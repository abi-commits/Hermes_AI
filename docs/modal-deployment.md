# Hermes Modal Deployment Guide

This document describes a realistic way to run Hermes on Modal without breaking the current application contract in [`hermes/main.py`](/home/abi/Documents/gen-ai/Hermes_AI/hermes/main.py).

It replaces the earlier draft that mixed aspirational architecture with non-working code.

---

## 1. Goal

Deploy Hermes so that:

- the FastAPI app runs as a Modal ASGI app
- Twilio still connects to the existing `/twilio/voice` webhook and `/stream/{call_sid}` WebSocket
- Gemini and Deepgram remain external managed services
- RAG continues to use the existing `ChromaRAGService`
- TTS either runs inside the API container exactly as it does today, or is split out behind a dedicated adapter layer

The current Hermes app already has a valid runtime entrypoint in [`hermes/main.py`](/home/abi/Documents/gen-ai/Hermes_AI/hermes/main.py). Any Modal deployment should preserve that behavior instead of re-implementing the app from scratch.

---

## 2. Current Hermes Runtime Contract

Hermes currently starts these shared services during FastAPI lifespan startup:

- `ChromaRAGService`
- `ChatterboxTTSService`
- `GeminiLLMService`
- `CallOrchestrator`
- module-level `connection_manager`

Source of truth:

- [`hermes/main.py`](/home/abi/Documents/gen-ai/Hermes_AI/hermes/main.py)
- [`hermes/websocket/handler.py`](/home/abi/Documents/gen-ai/Hermes_AI/hermes/websocket/handler.py)
- [`hermes/websocket/manager.py`](/home/abi/Documents/gen-ai/Hermes_AI/hermes/websocket/manager.py)
- [`hermes/core/call.py`](/home/abi/Documents/gen-ai/Hermes_AI/hermes/core/call.py)

Important constraints:

- The WebSocket route is mounted at `/stream/{call_sid}`.
- Twilio voice webhook responses point callers to `wss://<host>/stream/{CallSid}`.
- The active TTS interface is [`AbstractTTSService`](/home/abi/Documents/gen-ai/Hermes_AI/hermes/services/tts/base.py), which must provide:
  - `generate_stream(...)`
  - `generate(...)`
  - `sample_rate`
  - `set_executor(...)`
- [`Call._tts_task()`](/home/abi/Documents/gen-ai/Hermes_AI/hermes/core/call.py#L344) expects TTS streaming output to be 16-bit PCM at the model sample rate. Hermes itself performs resampling to 8 kHz and mu-law conversion before sending audio back to Twilio.

Because of that last point, a Modal TTS worker cannot simply return already-encoded Twilio audio unless Hermes is updated to use a different contract.

---

## 3. Recommended Deployment Shape

### Option A: Start with a single Modal ASGI app

This is the safest first deployment.

Architecture:

```text
Caller
  -> Twilio Voice Webhook
  -> Hermes ASGI app on Modal
      -> Deepgram
      -> Gemini
      -> Chroma / Chroma Cloud
      -> local Chatterbox TTS inside the same Modal container
```

Use this option when:

- getting Hermes onto Modal quickly matters more than GPU isolation
- expected traffic is modest
- you want minimum code change

Pros:

- reuses the current app almost unchanged
- avoids introducing a new remote TTS protocol
- simplest operational path

Cons:

- the API container also carries TTS runtime cost
- scaling API and TTS independently is not possible yet

### Option B: Split TTS into a separate Modal GPU service later

Use this only after adding a proper adapter that implements Hermes' `AbstractTTSService` interface while calling a remote Modal class/function.

Architecture:

```text
Caller
  -> Twilio Voice Webhook
  -> Hermes ASGI app on Modal
      -> remote Modal TTS adapter
          -> dedicated GPU-backed TTS worker
      -> Deepgram
      -> Gemini
      -> Chroma / Chroma Cloud
```

This requires real application code, not just deployment changes.

---

## 4. What Should Be Deployed on Modal

### FastAPI app

Deploy the existing Hermes app factory as an ASGI app.

Recommended wrapper:

```python
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .copy_local_file("pyproject.toml", "/app/pyproject.toml")
    .copy_local_dir("hermes", "/app/hermes")
    .copy_local_dir("config", "/app/config")
    .pip_install_from_pyproject("/app/pyproject.toml")
)

app = modal.App("hermes-api")

@modal.concurrent(max_inputs=100)
@app.function(image=image, cpu=4.0, memory=4096, container_idle_timeout=60)
@modal.asgi_app()
def fastapi_app():
    from hermes.main import create_app

    return create_app()
```

Notes:

- Keep imports inside the Modal function when they depend on copied project files.
- Copy project files before calling `pip_install_from_pyproject(...)`.
- Prefer reusing `create_app()` from [`hermes/main.py`](/home/abi/Documents/gen-ai/Hermes_AI/hermes/main.py) instead of rebuilding routers or lifespan logic in a second file.

### External services

These remain external to Modal:

- Twilio
- Deepgram
- Gemini
- Postgres / Neon
- Chroma Cloud, if used

### Storage

For production RAG, prefer a managed vector store or Chroma Cloud over a mutable ChromaDB-on-Volume design.

Modal Volumes are a better fit for:

- model caches
- read-mostly assets
- build artifacts
- seeded data that is not updated frequently at runtime

---

## 5. Environment and Secrets

Hermes settings come from [`config/settings.py`](/home/abi/Documents/gen-ai/Hermes_AI/config/settings.py). The Modal deployment should provide at least:

- `APP_ENV=production`
- `HOST=<public hostname if needed for downstream logic>`
- `PORT=8000` if explicitly used by tooling
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_PHONE_NUMBER`
- `TWILIO_TRANSFER_NUMBER` if transfer is enabled
- `DEEPGRAM_API_KEY`
- `GEMINI_API_KEY`
- `DATABASE_URL`
- `CHROMA_CLOUD_URL` and `CHROMA_CLOUD_API_KEY` if using Chroma Cloud
- `CHROMADB_HOST` and `CHROMADB_PORT` only if connecting to a separately hosted Chroma instance
- `CHATTERBOX_DEVICE=cuda` if the Modal container includes a GPU
- `CHATTERBOX_NUM_WORKERS`
- `CHATTERBOX_WATERMARK_KEY` if watermarking is enabled

Recommended Modal secret strategy:

- one secret for shared application credentials
- optional environment-specific secrets such as `hermes-prod`, `hermes-staging`

Example shape:

```python
secrets=[modal.Secret.from_name("hermes-prod")]
```

---

## 6. Twilio Integration Requirements

The current Twilio webhook implementation lives in [`hermes/api/twilio.py`](/home/abi/Documents/gen-ai/Hermes_AI/hermes/api/twilio.py).

Deployment requirements:

- expose `POST /twilio/voice`
- expose `POST /twilio/status`
- expose WebSocket `/stream/{call_sid}`
- preserve the existing `X-Forwarded-Host` behavior so the generated stream URL matches the public Modal hostname

Twilio should be configured to call:

- Voice webhook: `https://<modal-domain>/twilio/voice`
- Status callback: `https://<modal-domain>/twilio/status`

The WebSocket handler already supports these Twilio events and should be reused as-is:

- `connected`
- `start`
- `media`
- `dtmf`
- `mark`
- `clear`
- `stop`

Do not replace it with a reduced custom WebSocket loop unless the new implementation preserves the same behavior.

---

## 7. TTS on Modal

### Phase 1: Keep TTS local to the ASGI app

This is the only zero-surprise path with the current codebase.

Hermes already constructs `ChatterboxTTSService` during app startup in [`hermes/main.py`](/home/abi/Documents/gen-ai/Hermes_AI/hermes/main.py). If the Modal ASGI container has access to the right runtime dependencies and optionally a GPU, the existing TTS path continues to work.

Requirements:

- include Chatterbox dependencies in the Modal image
- give the function a GPU only if you want Chatterbox to run on GPU
- keep `CHATTERBOX_DEVICE` aligned with the container resources

### Phase 2: Move TTS to a separate GPU worker

If TTS needs to scale separately, add a new adapter in Hermes, for example:

- `hermes/services/tts/modal_remote.py`

That adapter must:

- implement `AbstractTTSService`
- expose `sample_rate`
- call a remote Modal method or generator
- return PCM bytes in the same format Hermes expects today
- support cancellation or interruption semantics well enough for barge-in behavior

Until that adapter exists, the deployment guide should not claim that `modal.Cls.lookup(...)` can be dropped directly into `ServiceBundle` as `tts_service`.

---

## 8. Autoscaling and Concurrency

Modal handles scaling at the function level. Keep the deployment model simple:

- use `@modal.concurrent(...)` for input concurrency
- tune container CPU and memory based on observed websocket load
- increase warm capacity only when cold starts become a real issue

Practical guidance:

- start with one ASGI deployment
- measure concurrent call count, startup latency, and memory use
- scale conservatively because each active call holds live state in memory

Do not document manual container-count management unless there is a real operational tool in the repo that uses supported Modal APIs.

---

## 9. Monitoring

Hermes already exposes:

- health endpoints
- Prometheus-style metrics routes
- structured logs via `structlog`

Relevant code:

- [`hermes/api/health.py`](/home/abi/Documents/gen-ai/Hermes_AI/hermes/api/health.py)
- [`hermes/api/metrics.py`](/home/abi/Documents/gen-ai/Hermes_AI/hermes/api/metrics.py)
- [`hermes/main.py`](/home/abi/Documents/gen-ai/Hermes_AI/hermes/main.py)

Recommended production checks:

- `GET /health`
- `GET /metrics/...` routes already registered by the app
- Twilio webhook success rate
- active websocket count
- Deepgram connection errors
- Gemini latency
- TTS latency

Prefer exporting Hermes' own metrics rather than inventing separate Modal-only health scripts first.

---

## 10. Implementation Checklist

### Minimum viable Modal deployment

1. Create a Modal image that installs Hermes and its runtime dependencies.
2. Wrap `create_app()` from [`hermes/main.py`](/home/abi/Documents/gen-ai/Hermes_AI/hermes/main.py) in `@modal.asgi_app()`.
3. Attach the required Modal secret containing Hermes environment variables.
4. Deploy the app and verify:
   - `POST /twilio/voice`
   - `POST /twilio/status`
   - `GET /health`
   - WebSocket `/stream/{call_sid}`
5. Point the Twilio phone number or webhook to the Modal URL.
6. Run an end-to-end inbound-call test.

### Second phase

1. Decide whether Chroma should remain external or move to Chroma Cloud.
2. Measure whether in-container TTS is sufficient.
3. If not, build a dedicated remote TTS adapter that satisfies `AbstractTTSService`.
4. Only then split TTS into a separate Modal GPU workload.

---

## 11. Non-Goals

This guide intentionally does not claim that Hermes already has:

- a production-ready `modal/` package
- TOML-driven Modal configuration consumed by Modal itself
- automatic geo-routing across multiple Modal regions
- a drop-in remote TTS implementation compatible with current Hermes runtime contracts
- a supported autoscaling control plane implemented in this repo

Those can be added later, but they should be documented as future work, not as if they already exist.

---

## 12. Suggested Next Code Changes

If we decide to make Modal a first-class deployment target, the next concrete code changes should be:

1. Add `modal/app.py` that wraps `hermes.main:create_app`.
2. Add a small deployment README with exact `modal deploy` commands.
3. Add `hermes/services/tts/modal_remote.py` only if we want separate GPU-backed TTS.
4. Add deployment smoke tests for `/twilio/voice`, `/health`, and the WebSocket handshake path.

That sequence keeps the docs honest and the implementation incremental.
