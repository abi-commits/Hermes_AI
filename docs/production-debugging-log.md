# Production Debugging & Architecture Refinement Log

This document summarizes the technical challenges discovered during the production verification of **Hermes AI** (March 2026) and the engineering solutions implemented to achieve a robust, low-latency streaming architecture.

## 1. WebSocket Handshake Timeouts
*   **The Problem:** Initial connections to the production WebSocket frequently timed out or took over 90 seconds to establish (returning `101 Switching Protocols`). 
*   **Root Cause:** The FastAPI `lifespan` and the WebSocket handler were performing synchronous, heavy initialization (loading RAG models, initializing Gemini, connecting to Deepgram). Modal's Web Gateway would time out before the application signaled it was "Ready."
*   **The Solution:** 
    *   **Instant Acceptance:** The WebSocket now calls `await websocket.accept()` immediately upon connection.
    *   **Background Lifespan:** Heavy service initialization was moved into an asynchronous background task within the FastAPI `lifespan`, allowing the web server to start in milliseconds.
    *   **Ready-Check Polling:** The WebSocket handler now polls for the `orchestrator` to be ready (up to 30s) before processing events, ensuring services are live without blocking the handshake.

## 2. The "Silent Stream" (Generator Mismatch)
*   **The Problem:** Handshakes succeeded, and logs showed synthesis was requested, but no audio chunks arrived at the client.
*   **Root Cause:** A mismatch between how the API consumed the remote Modal stream and how the Modal SDK yielded generators. The API was often waiting for a "bucket" (complete object) instead of a "hose" (streaming chunks).
*   **The Solution:**
    *   **Normalization:** Implemented `_call_remote_gen` in `ModalRemoteTTSService` to wrap all remote calls.
    *   **Explicit Streaming:** Updated the TTS worker to use `@modal.method(is_generator=True)` and return `AsyncIterator[bytes]`.
    *   **Internal Buffering:** Added a normalization loop that detects if the returned object is an `AsyncIterator` and yields chunks explicitly.

## 3. Production Dependency Crashes
*   **The Problem:** Logs showed `ImportError` and `Deepgram SDK not installed`, despite the library being in the dependency list.
*   **Root Cause:** 
    1.  **Image Caching:** Modal's builder was using cached layers that didn't include the newly added `deepgram-sdk`.
    2.  **Legacy Code:** A refactoring of the logging utility removed the `JSONFormatter` class, but a reference remained in `hermes/utils/__init__.py`, causing a crash on boot.
*   **The Solution:**
    *   **Cache Busting:** Added a `CACHE_BUST` timestamp comment to the Modal image definition to force a clean rebuild of all layers.
    *   **Import Cleanup:** Hard-deleted the obsolete references in `utils/__init__.py`.
    *   **Explicit Dependency:** Moved the SDK installation to the very top of the `pip_install` list to ensure it is prioritized.

## 4. Disconnected Observability
*   **The Problem:** Debugging the "Relay Race" between STT, LLM, and TTS was difficult because logs were scattered and used different formats.
*   **Root Cause:** A mix of standard Python `logging` and `structlog` without a unified configuration or shared context (like `call_sid`).
*   **The Solution:**
    *   **Unified Logging:** Created a centralized `configure_logging` in `hermes/utils/logging.py` that pipes all library logs through `structlog`.
    *   **Environment Awareness:** Logs now automatically switch to **JSON format** when `APP_ENV=production` (for cloud parsing) and **Console format** in development (for readability).
    *   **Context Binding:** The `Call` object now binds the `call_sid` to its logger at creation. Every log line across the entire pipeline for that specific call now carries the same ID, enabling instant filtering.

## 5. High Latency Cold Starts
*   **The Problem:** First-call users experienced silence for 60+ seconds while the GPU worker loaded the model.
*   **Root Cause:** `min_containers=0` meant the GPU had to provision and load the "AI brain" from scratch for every new session.
*   **The Solution:** 
    *   **Warm Infrastructure:** Set `min_containers=1` for both the API and TTS server in production.
    *   **Priming:** Updated the diagnostic tools to send silent priming audio to wake up the STT pipeline immediately upon connection.

---

### Final Verified State
The system has been verified locally using a **Python 3.11** environment with **Remote Modal TTS**. The logic successfully handles the Twilio handshake, prioritizes `customParameters` for greetings, and correctly routes data through the background orchestrator. 

**Current Version:** 0.1.0  
**Deployment Profile:** Warm (min_containers=1)  
**Logging Strategy:** Unified Structured JSON  
