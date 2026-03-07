# Hermes Modal Deployment

Production deployment entrypoint:

- [`modal_deploy/app.py`](/home/abi/Documents/gen-ai/Hermes_AI/modal_deploy/app.py)
- [`modal_deploy/tts.py`](/home/abi/Documents/gen-ai/Hermes_AI/modal_deploy/tts.py)

## Install

```bash
uv sync --extra all --extra modal
```

## Required Modal secret

Create a Modal secret that includes the Hermes production environment variables, for example:

- `APP_ENV=production`
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_PHONE_NUMBER`
- `DEEPGRAM_API_KEY`
- `GEMINI_API_KEY`
- `CHROMA_CLOUD_URL`
- `CHROMA_CLOUD_API_KEY`
- `DATABASE_URL`

By default the deployment expects the secret name `hermes-prod`.

Override with:

```bash
export HERMES_MODAL_SECRET_NAMES=hermes-prod
```

## Deploy

```bash
uv run --extra modal modal deploy modal_deploy/tts.py
make modal-deploy-prod
```

Or directly:

```bash
uv run --extra modal modal deploy modal_deploy/tts.py
HERMES_MODAL_ENV=production uv run --extra modal modal deploy modal_deploy/app.py
```

If you want Hermes to use the dedicated remote worker, include these env vars in the Modal app secret:

- `TTS_PROVIDER=modal_remote`
- `MODAL_TTS_APP_NAME=hermes-tts`
- `MODAL_TTS_CLASS_NAME=RemoteChatterboxTTSWorker`
- `MODAL_TTS_SAMPLE_RATE=24000`

## Smoke test the remote TTS path

After deploying the TTS worker, validate the adapter end-to-end before switching live traffic:

```bash
TTS_PROVIDER=modal_remote make modal-smoke-tts
```

This verifies:

- Modal class lookup succeeds
- full-audio generation returns non-empty PCM bytes
- streaming generation returns non-empty PCM chunks
- chunk framing matches Hermes' expected 16-bit PCM contract

## Optional tuning

Environment variables read by [`modal_deploy/config.py`](/home/abi/Documents/gen-ai/Hermes_AI/modal_deploy/config.py):

- `HERMES_MODAL_APP_NAME`
- `HERMES_MODAL_ENDPOINT_LABEL`
- `HERMES_MODAL_CPU`
- `HERMES_MODAL_MEMORY_MB`
- `HERMES_MODAL_CONCURRENCY`
- `HERMES_MODAL_TIMEOUT_S`
- `HERMES_MODAL_GPU`
- `HERMES_MODAL_MIN_CONTAINERS`
- `HERMES_MODAL_SCALEDOWN_WINDOW_S`
- `HERMES_MODAL_REGION`
- `HERMES_MODAL_SECRET_NAMES`
- `HERMES_MODAL_MODEL_CACHE_VOLUME`
- `HERMES_MODAL_MODEL_CACHE_PATH`
- `HERMES_MODAL_TTS_APP_NAME`
- `HERMES_MODAL_TTS_CLASS_NAME`
- `HERMES_MODAL_TTS_GPU`
- `HERMES_MODAL_TTS_CPU`
- `HERMES_MODAL_TTS_MEMORY_MB`
- `HERMES_MODAL_TTS_TIMEOUT_S`
- `HERMES_MODAL_TTS_CONCURRENCY`
- `HERMES_MODAL_TTS_MIN_CONTAINERS`
- `HERMES_MODAL_TTS_SCALEDOWN_WINDOW_S`
- `HERMES_MODAL_TTS_LOCAL_WORKERS`
- `HERMES_MODAL_TTS_RUNTIME_DEVICE`

## Local validation

```bash
make modal-preflight
```
