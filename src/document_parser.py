"""Document Parser module for Pipeline Documents.

Two-layer parsing strategy:
1. Fast path — deterministic JSON/YAML parsing for structured input.
2. LLM path — uses a language model to extract data_source, transformations,
   and architecture from free-form or messy input, and to classify whether
   user input is a new pipeline request vs. a follow-up question.

Provides pure functions for validation and pretty-printing, plus an
async LLM-backed extraction function.
"""

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

S3_URI_PATTERN = re.compile(r"^s3://[a-z0-9][a-z0-9.\-]{1,61}[a-z0-9]/\S+$")

REQUIRED_SECTIONS = ("data_source", "transformations", "architecture")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class InputIntent(str, Enum):
    """Classified intent of user input."""
    NEW_PIPELINE = "new_pipeline"
    FOLLOWUP_QUESTION = "followup_question"
    MODIFICATION = "modification"
    FULL_REGENERATION = "full_regeneration"


@dataclass
class ParsedDocument:
    """Structured representation of a parsed Pipeline Document."""
    data_source: dict[str, Any]
    transformations: list[dict[str, Any]]
    architecture: dict[str, Any]
    raw_content: str  # original input for re-parse on full regeneration


@dataclass
class ParseError:
    """Describes a parsing or validation error."""
    message: str
    line: int | None = None
    column: int | None = None


@dataclass
class ClassifiedInput:
    """Result of LLM-based intent classification."""
    intent: InputIntent
    parsed_document: ParsedDocument | None = None
    errors: list[ParseError] | None = None
    raw_message: str = ""


# ---------------------------------------------------------------------------
# Deterministic parsing (fast path for clean JSON/YAML)
# ---------------------------------------------------------------------------

def _try_parse_structured(raw: str) -> dict | None:
    """Attempt JSON then YAML parse. Returns dict or None."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        data = yaml.safe_load(raw)
        if isinstance(data, dict):
            return data
    except yaml.YAMLError:
        pass
    return None


def validate_document(data: dict) -> list[ParseError]:
    """Validate required sections and S3 URI format."""
    errors: list[ParseError] = []
    for section in REQUIRED_SECTIONS:
        if section not in data:
            errors.append(ParseError(message=f"Missing required section: '{section}'"))

    ds = data.get("data_source")
    if isinstance(ds, dict):
        uri = ds.get("uri", "")
        if uri and not S3_URI_PATTERN.match(uri):
            errors.append(ParseError(
                message=f"Malformed S3 URI: '{uri}' — expected s3://<bucket>/<key>"
            ))
    return errors


def parse_pipeline_document(raw: str) -> ParsedDocument | list[ParseError]:
    """Parse a raw JSON or YAML string into a ParsedDocument.

    Auto-detects format: tries JSON first, then YAML fallback.
    Returns ParsedDocument on success, or a list of ParseError on failure.
    """
    # --- Try JSON first ---
    data = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as json_err:
        # --- Try YAML fallback ---
        try:
            data = yaml.safe_load(raw)
            if not isinstance(data, dict):
                return [ParseError(
                    message="YAML parsed but result is not a mapping (dict)",
                )]
        except yaml.YAMLError as yaml_err:
            # Both failed — report the YAML error (since JSON already failed)
            line, col = None, None
            if hasattr(yaml_err, "problem_mark") and yaml_err.problem_mark:
                line = yaml_err.problem_mark.line + 1
                col = yaml_err.problem_mark.column + 1
            return [
                ParseError(
                    message=f"JSON parse error: {json_err.msg}",
                    line=json_err.lineno,
                    column=json_err.colno,
                ),
                ParseError(
                    message=f"YAML parse error: {yaml_err}",
                    line=line,
                    column=col,
                ),
            ]

    # --- Validate ---
    errors = validate_document(data)
    if errors:
        return errors

    return ParsedDocument(
        data_source=data["data_source"],
        transformations=data["transformations"],
        architecture=data["architecture"],
        raw_content=raw,
    )


def pretty_print(doc: ParsedDocument, fmt: str = "yaml") -> str:
    """Pretty-print a ParsedDocument back to YAML or JSON string."""
    payload = {
        "data_source": doc.data_source,
        "transformations": doc.transformations,
        "architecture": doc.architecture,
    }
    if fmt == "json":
        return json.dumps(payload, indent=2, default=str)
    return yaml.dump(payload, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# LLM-based extraction and intent classification
# ---------------------------------------------------------------------------

_CLASSIFY_AND_EXTRACT_PROMPT = """\
You are a pipeline document classifier and extractor.

Given user input, do TWO things:

1. **Classify intent** — determine if the input is:
   - "new_pipeline": The user is submitting a new pipeline definition \
(structured document, free-form description of a data pipeline, or any input \
that contains data source location, transformation requirements, and architecture).
   - "followup_question": The user is asking a question about a previously \
generated pipeline (e.g. "why did you choose Glue?", "explain the VPC setup").
   - "modification": The user wants to change part of an existing pipeline \
(e.g. "redo the IaC with a different VPC", "change the output format to parquet").
   - "full_regeneration": The user wants to start over completely \
(e.g. "redo from scratch", "regenerate everything").

2. **Extract sections** (only when intent is "new_pipeline"):
   - `data_source`: An object with at minimum a `uri` field (S3 URI). \
Include `format` and `options` if mentioned.
   - `transformations`: A list of transformation steps. Each should have \
`name` and `description`. Include `output_location` if mentioned.
   - `architecture`: An object describing compute, storage, networking, \
and any other infrastructure details.

Respond with ONLY a JSON object (no markdown fences) in this exact shape:
{
  "intent": "<new_pipeline|followup_question|modification|full_regeneration>",
  "data_source": { ... } | null,
  "transformations": [ ... ] | null,
  "architecture": { ... } | null,
  "extraction_notes": "any notes about assumptions made, or null"
}

If intent is NOT "new_pipeline", set data_source, transformations, and \
architecture to null.

If the input is a new pipeline but is missing some sections, extract what \
you can and set missing sections to null. Add a note in extraction_notes.
"""


async def classify_and_extract(
    raw: str,
    model: BaseChatModel,
) -> ClassifiedInput:
    """Use an LLM to classify user intent and extract pipeline sections.

    Strategy:
    1. If the input is clean JSON/YAML with all required sections, skip the
       LLM call entirely (fast path).
    2. Otherwise, call the LLM to classify intent and extract sections from
       free-form or messy input.
    """
    # Fast path: try deterministic parse first
    structured = _try_parse_structured(raw)
    if structured is not None:
        errors = validate_document(structured)
        if not errors:
            doc = ParsedDocument(
                data_source=structured["data_source"],
                transformations=structured["transformations"],
                architecture=structured["architecture"],
                raw_content=raw,
            )
            return ClassifiedInput(
                intent=InputIntent.NEW_PIPELINE,
                parsed_document=doc,
                raw_message=raw,
            )

    # LLM path: classify and extract
    response = await model.ainvoke([
        SystemMessage(content=_CLASSIFY_AND_EXTRACT_PROMPT),
        HumanMessage(content=raw),
    ])

    # Parse LLM response
    try:
        result = json.loads(response.content)
    except (json.JSONDecodeError, TypeError):
        return ClassifiedInput(
            intent=InputIntent.FOLLOWUP_QUESTION,
            errors=[ParseError(message="Failed to parse LLM extraction response")],
            raw_message=raw,
        )

    intent_str = result.get("intent", "followup_question")
    try:
        intent = InputIntent(intent_str)
    except ValueError:
        intent = InputIntent.FOLLOWUP_QUESTION

    # For non-pipeline intents, return classification only
    if intent != InputIntent.NEW_PIPELINE:
        return ClassifiedInput(intent=intent, raw_message=raw)

    # Build ParsedDocument from LLM extraction
    ds = result.get("data_source")
    transforms = result.get("transformations")
    arch = result.get("architecture")

    missing: list[ParseError] = []
    if ds is None:
        missing.append(ParseError(message="LLM could not extract 'data_source' from input"))
    if transforms is None:
        missing.append(ParseError(message="LLM could not extract 'transformations' from input"))
    if arch is None:
        missing.append(ParseError(message="LLM could not extract 'architecture' from input"))

    if missing:
        notes = result.get("extraction_notes")
        if notes:
            missing.append(ParseError(message=f"Extraction notes: {notes}"))
        return ClassifiedInput(
            intent=InputIntent.NEW_PIPELINE,
            errors=missing,
            raw_message=raw,
        )

    if not isinstance(transforms, list):
        transforms = [transforms]

    doc = ParsedDocument(
        data_source=ds,
        transformations=transforms,
        architecture=arch,
        raw_content=raw,
    )

    # Validate the extracted document (S3 URI check, etc.)
    validation_errors = validate_document({
        "data_source": ds,
        "transformations": transforms,
        "architecture": arch,
    })
    if validation_errors:
        return ClassifiedInput(
            intent=InputIntent.NEW_PIPELINE,
            parsed_document=doc,
            errors=validation_errors,
            raw_message=raw,
        )

    return ClassifiedInput(
        intent=InputIntent.NEW_PIPELINE,
        parsed_document=doc,
        raw_message=raw,
    )


# ---------------------------------------------------------------------------
# Apply tracing wrappers
# ---------------------------------------------------------------------------

from src.tracing.parser import traced_classify_and_extract, traced_parse_pipeline_document

parse_pipeline_document = traced_parse_pipeline_document(parse_pipeline_document)
classify_and_extract = traced_classify_and_extract(classify_and_extract)


# ---------------------------------------------------------------------------
# LangChain tool wrapper (for orchestrator tool registration)
# ---------------------------------------------------------------------------

from langchain_core.tools import tool


@tool
def parse_document_tool(raw_content: str) -> str:
    """Parse and validate a Pipeline Document (JSON or YAML).

    Returns a pretty-printed YAML summary on success,
    or validation error details on failure.
    """
    result = parse_pipeline_document(raw_content)
    if isinstance(result, list):  # errors
        return "PARSE ERRORS:\n" + "\n".join(
            f"- {e.message}" + (f" (line {e.line})" if e.line else "")
            for e in result
        )
    return "PARSE SUCCESS:\n" + pretty_print(result)
