from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_CONFIGURED = False
_SERVICE_NAME = "self-improving-coding-agent"


def configure_tracing() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    enabled = _pick_bool("TRACING_ENABLED", False)
    if not enabled:
        _CONFIGURED = True
        return

    endpoint = _pick_endpoint()
    resource = Resource.create(
        {
            "service.name": os.getenv("OTEL_SERVICE_NAME", _SERVICE_NAME),
            "service.version": os.getenv("OTEL_SERVICE_VERSION", "0.1.0"),
        }
    )
    provider = TracerProvider(resource=resource)
    if endpoint:
        exporter = OTLPSpanExporter(endpoint=endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _CONFIGURED = True


def get_tracer():
    return trace.get_tracer(os.getenv("OTEL_SERVICE_NAME", _SERVICE_NAME))


@contextmanager
def span(name: str, attributes: Optional[Dict[str, object]] = None) -> Iterator[object]:
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span_obj:
        if attributes:
            for key, value in attributes.items():
                span_obj.set_attribute(key, value)
        yield span_obj


def _pick_endpoint() -> Optional[str]:
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or os.getenv("PHOENIX_OTLP_ENDPOINT")
    if endpoint:
        endpoint = endpoint.rstrip("/")
        if not endpoint.endswith("/v1/traces"):
            endpoint = f"{endpoint}/v1/traces"
        return endpoint
    if _pick_bool("TRACING_DEFAULT_PHOENIX", False):
        return "http://localhost:6006/v1/traces"
    return None


def _pick_bool(env_key: str, default: bool) -> bool:
    raw = os.getenv(env_key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
