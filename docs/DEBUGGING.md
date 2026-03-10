# Troubleshooting & Debugging

This guide details the technical resolutions for common issues encountered in the Hermes AI production environment.

## 1. WebSocket & Handshake Issues

### **Handshake Timeout (101 Switching Protocols delay)**
*   **Symptom:** Connection takes >10s to establish or fails with a timeout.
*   **Resolution:** Ensure the `lifespan` in `hermes/main.py` is non-blocking. Heavy service initialization (RAG/LLM) must run in an asynchronous background task.
*   **Validation:** Logs should show `orchestrator_attached_and_ready` within 1 second of boot.

### **Call SID Mismatch (4001 Error)**
*   **Symptom:** WebSocket closes immediately with code 4001.
*   **Resolution:** The `callSid` in the Twilio `start` event JSON **must** exactly match the `{call_sid}` in the WebSocket URL.
*   **Check:** Verify your diagnostic script or Twilio TwiML template.

## 2. STT (Deepgram) Failures

### **Rejection with 400 Bad Request**
*   **Symptom:** Deepgram connection fails instantly.
*   **Cause 1 (Encoding):** You are sending `mulaw` but requested `linear16`.
    *   *Fix:* Set `encoding="mulaw"` and `sample_rate=8000` in `DeepgramSTTService`.
*   **Cause 2 (Parameters):** Unsupported parameters like `utterance_end_ms` can trigger rejections in Mu-law mode.
    *   *Fix:* Remove `utterance_end_ms` and ensure `channels=1` is explicitly set.

## 3. TTS (Modal GPU) Issues

### **The "Silent Stream" (No chunks received)**
*   **Symptom:** Handshake succeeds, LLM generates text, but no audio arrives.
*   **Resolution:** Verify the Modal consumption pattern in `ModalRemoteTTSService`. 
    *   *Fix:* Use the pattern `instance.method.remote_gen.aio(...)` for reliable streaming. 
    *   *Cache:* Ensure the remote instance is cached (`self._remote_instance`) to avoid instantiation overhead on every sentence.

### **High Latency / Slow Response**
*   **Symptom:** First audio chunk takes >30 seconds.
*   **Resolution:** 
    1.  Set `min_containers=1` in `modal_deploy/config.py`.
    2.  Ensure model weights are cached in a **Modal Volume** (`/cache/hf`).

## 4. LLM & Logic Crashes

### **ConversationTurn TypeError**
*   **Symptom:** `llm_task_error` in logs regarding unexpected arguments.
*   **Resolution:** Ensure the `ConversationTurn` dataclass in `hermes/models/call.py` includes the `interrupted: bool` field to support barge-in tracking.

---

## Useful Debugging Commands

### **View Live Production Logs**
```bash
modal app logs hermes-api
```

### **Check Service Readiness**
```bash
curl https://<user>--hermes-api.modal.run/ready
```

### **Test LLM Streaming (No Voice Required)**
```bash
uv run scripts/diagnose_prod_stream.py --prompt "Say hello"
```

---
**Status:** Debug-Verified
