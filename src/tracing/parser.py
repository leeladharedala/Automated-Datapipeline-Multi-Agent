"""Document parser tracing wrappers.

Wraps ``parse_pipeline_document`` and ``classify_and_extract`` to produce
spans that capture parsing path, format detection, and success/failure
attributes.
"""

from __future__ import annotations

import functools
import json
from typing import Any, Callable

from opentelemetry.trace import StatusCode

from src.tracing.provider import get_tracer
from src.tracing.utils import record_exception


def traced_parse_pipeline_document(original_fn: Callable) -> Callable:
    """Wrap ``parse_pipeline_document`` in a ``"parse:pipeline_document"`` span.

    Attributes set:
        ``parse.input_size`` – byte length of the raw input.
        ``parse.format_detected`` – ``"json"``, ``"yaml"``, or ``"unknown"``.
        ``parse.success`` – whether parsing produced a ``ParsedDocument``.
        ``parse.section_count`` – number of top-level sections (on success).
        ``parse.error_count`` – number of ``ParseError`` objects (on failure).
    """

    @functools.wraps(original_fn)
    def wrapper(raw: str, *args: Any, **kwargs: Any) -> Any:
        tracer = get_tracer("multi-agent-pipeline")
        with tracer.start_as_current_span("parse:pipeline_document") as span:
            span.set_attribute("parse.input_size", len(raw.encode("utf-8")))
            span.set_attribute("parse.format_detected", _detect_format(raw))
            try:
                result = original_fn(raw, *args, **kwargs)
                _set_result_attributes(span, result)
                span.set_status(StatusCode.OK)
                return result
            except Exception as exc:
                record_exception(span, exc)
                raise

    return wrapper


def traced_classify_and_extract(original_fn: Callable) -> Callable:
    """Wrap ``classify_and_extract`` in a ``"parse:classify_and_extract"`` span.

    Attributes set:
        ``parse.path`` – ``"fast"`` when deterministic parsing succeeds,
        ``"llm"`` when the LLM fallback is used.

    When the LLM path is taken a child span ``"parse:llm_extraction"`` is
    created with ``parse.intent`` set to the classified intent value.
    """

    @functools.wraps(original_fn)
    async def wrapper(raw: str, model: Any, *args: Any, **kwargs: Any) -> Any:
        tracer = get_tracer("multi-agent-pipeline")
        with tracer.start_as_current_span("parse:classify_and_extract") as span:
            try:
                result = await original_fn(raw, model, *args, **kwargs)

                # Determine which path was taken: if the result has a
                # parsed_document with raw_content matching the input AND
                # no LLM would have been needed (clean structured input),
                # it's the fast path.  The fast path returns
                # intent=NEW_PIPELINE with a parsed_document and no errors
                # when _try_parse_structured + validate_document both pass.
                path = _infer_path(raw, result)
                span.set_attribute("parse.path", path)

                if path == "llm":
                    intent_value = getattr(
                        getattr(result, "intent", None), "value", str(getattr(result, "intent", ""))
                    )
                    with tracer.start_as_current_span("parse:llm_extraction") as llm_span:
                        llm_span.set_attribute("parse.intent", intent_value)
                        llm_span.set_status(StatusCode.OK)

                span.set_status(StatusCode.OK)
                return result
            except Exception as exc:
                span.set_attribute("parse.path", "llm")
                record_exception(span, exc)
                raise

    return wrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_format(raw: str) -> str:
    """Detect whether *raw* looks like JSON, YAML, or unknown."""
    stripped = raw.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            json.loads(stripped)
            return "json"
        except (json.JSONDecodeError, ValueError):
            pass
    # Could still be YAML even if JSON failed
    if ":" in stripped:
        return "yaml"
    return "unknown"


def _set_result_attributes(span: Any, result: Any) -> None:
    """Set success/failure attributes on the parse span based on the result."""
    from src.document_parser import ParsedDocument

    if isinstance(result, ParsedDocument):
        span.set_attribute("parse.success", True)
        # Count top-level sections present in the document
        section_count = sum(
            1
            for attr in ("data_source", "transformations", "architecture")
            if getattr(result, attr, None) is not None
        )
        span.set_attribute("parse.section_count", section_count)
    elif isinstance(result, list):
        span.set_attribute("parse.success", False)
        span.set_attribute("parse.error_count", len(result))
    else:
        span.set_attribute("parse.success", False)


def _infer_path(raw: str, result: Any) -> str:
    """Infer whether the fast or LLM path was taken.

    The fast path succeeds only when ``_try_parse_structured`` returns a
    valid dict *and* ``validate_document`` finds no errors.  In that case
    ``classify_and_extract`` returns a ``ClassifiedInput`` with
    ``intent=NEW_PIPELINE``, a ``parsed_document``, and no ``errors``.
    Any other outcome means the LLM path was used.
    """
    from src.document_parser import InputIntent

    intent = getattr(result, "intent", None)
    parsed_doc = getattr(result, "parsed_document", None)
    errors = getattr(result, "errors", None)

    if (
        intent == InputIntent.NEW_PIPELINE
        and parsed_doc is not None
        and not errors
    ):
        # Could still be LLM-extracted.  The fast path builds the
        # ParsedDocument from _try_parse_structured which always sets
        # raw_content = raw.  The LLM path also sets raw_content = raw,
        # so we can't distinguish purely from the result.  Instead, try
        # the deterministic parse ourselves to see if it would succeed.
        try:
            structured = json.loads(raw)
            if isinstance(structured, dict):
                from src.document_parser import REQUIRED_SECTIONS
                if all(k in structured for k in REQUIRED_SECTIONS):
                    return "fast"
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            import yaml
            structured = yaml.safe_load(raw)
            if isinstance(structured, dict):
                from src.document_parser import REQUIRED_SECTIONS
                if all(k in structured for k in REQUIRED_SECTIONS):
                    return "fast"
        except Exception:
            pass
        return "llm"

    return "llm"
