import os
import logging

logger = logging.getLogger("xray_api")

OTEL_ENABLED = os.getenv("OTEL_ENABLED", "false").lower() == "true"
OTEL_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "xray-report-api")
OTEL_EXPORTER_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")


def configure_telemetry(app=None):
    if not OTEL_ENABLED:
        logger.info("OpenTelemetry disabled (set OTEL_ENABLED=true to enable).")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        resource = Resource(attributes={SERVICE_NAME: OTEL_SERVICE_NAME})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=OTEL_EXPORTER_ENDPOINT, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        if app is not None:
            FastAPIInstrumentor.instrument_app(app)

        logger.info("OpenTelemetry configured → %s", OTEL_EXPORTER_ENDPOINT)

    except ImportError:
        logger.warning(
            "OpenTelemetry packages not installed. "
            "Install: opentelemetry-sdk opentelemetry-exporter-otlp "
            "opentelemetry-instrumentation-fastapi"
        )
