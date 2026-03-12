# Development Guide

This guide covers local development, testing, and diagnostics for Hermes AI.

## 1. Local Environment Setup

We use `uv` for high-speed dependency management.

```bash
# 1. Install dependencies
uv sync --extra all

# 2. Configure environment
cp .env.example .env
# (Fill in your API keys)
```

## 2. Running the Local Server

You can run the full API and WebSocket orchestrator on your machine.

```bash
# Recommended: Use Modal for TTS during local development
TTS_PROVIDER=modal_remote uv run uvicorn hermes.main:app --host 0.0.0.0 --port 8000
```

## 3. End-to-End Diagnostics

We have a specialized diagnostic script to verify the entire Ear-Brain-Mouth pipeline.

### **Local Test**
Verify your local code logic while using the remote GPU for speech.
```bash
uv run scripts/diagnose_prod_stream.py --local --prompt "Verify the full pipeline."
```

### **Production Test**
Verify the live Modal deployment.
```bash
uv run scripts/diagnose_prod_stream.py --prompt "Is the production system live?"
```

## 4. Structured Logging Standards

All logs are emitted via `structlog`. Follow these rules for consistency:

*   **Snake Case Events:** Use lowercase with underscores (e.g., `call_created`).
*   **Key-Value Context:** Pass dynamic data as arguments, not in the string.
*   **Log Binding:** Always use the bound logger from the `Call` object if available to ensure the `call_sid` is tracked.

**Good Example:**
```python
logger.info("tts_audio_sent", chunks=chunk_count, latency_ms=850)
```

## 5. Telemetry & Metrics

The system exposes a Prometheus-compatible `/metrics` endpoint.
*   **Local:** `http://localhost:8000/metrics`
*   **Production:** `https://<user>--hermes-api.modal.run/metrics`

Key metrics to watch:
*   `hermes_calls_active`: Count of concurrent callers.
*   `hermes_audio_bytes_total`: Data throughput.


