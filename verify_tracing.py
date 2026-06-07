"""
Build-time tracing verification script.
Run inside the Docker image after `COPY src/` to catch import errors early.
"""
import sys
sys.path.insert(0, '/app')

print("Verifying tracing module imports...")

from src.tracing.provider import setup_tracing, get_tracer, shutdown_tracing
print("  tracing.provider OK")

from src.tracing.utils import record_exception, traced, traced_span
print("  tracing.utils OK")

from src.tracing.parser import traced_classify_and_extract, traced_parse_pipeline_document
print("  tracing.parser OK")

from src.tracing.graphs import instrument_graph
print("  tracing.graphs OK")

from src.tracing import setup_tracing  # noqa: F811 — re-import via __init__
print("  tracing __init__ OK")

# Run setup and create a span to ensure the provider initialises cleanly.
setup_tracing()
print("  setup_tracing() OK")

tracer = get_tracer("build-verify")
span = tracer.start_span("build_verify_span")
span.set_attribute("build", True)
span.end()
print("  span creation OK")

print("All tracing imports verified. Build OK.")
