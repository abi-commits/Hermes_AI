# Deployment Guide

Hermes AI is designed to run on **Modal**, providing serverless GPU scaling and simplified orchestration.

## 1. Prerequisites

*   **Modal Account:** Sign up at [modal.com](https://modal.com).
*   **API Keys:**
    *   `GOOGLE_API_KEY`: For Gemini 2.5 Flash.
    *   `DEEPGRAM_API_KEY`: For Nova-2 STT.
    *   `TWILIO_ACCOUNT_SID` & `TWILIO_AUTH_TOKEN`: For telephony integration.

## 2. Environment Configuration

### **Modal Secrets**
Create a Modal Secret named `hermes-prod` containing the API keys above. You can do this in the Modal Dashboard or via the CLI:
```bash
modal secret create hermes-prod GOOGLE_API_KEY=... DEEPGRAM_API_KEY=...
```

### **Local Environment**
Copy `.env.example` to `.env` and fill in your local development keys.

## 3. Production Deployment

The system is deployed in two stages using the provided `Makefile`:

### **Stage 1: TTS GPU Worker**
Deploys the Chatterbox TTS model to a serverless GPU cluster.
```bash
make modal-deploy-tts
```
*   **Infrastructure:** NVIDIA T4 GPU.
*   **Caching:** Uses a Modal Volume (`hermes-model-cache`) to persist model weights across cold starts.

### **Stage 2: Main API**
Deploys the FastAPI server and WebSocket orchestrator.
```bash
make modal-deploy-prod
```
*   **Inbound:** `https://<user>--hermes-api.modal.run`
*   **Telephony:** Point your Twilio Webhook to this URL.

## 4. Cold Start Mitigation

To ensure high performance for real-world callers, we use **Warm Infrastructure** settings:
*   `min_containers=1` is set for the API.
*   `min_containers=1` is recommended for the TTS worker during high-traffic hours.
*   **Volume Caching:** Model weights are stored in `/cache/hf` to skip download times.

## 5. Telephony Setup (Twilio)

1.  Buy a Twilio Number.
2.  Configure the **Voice Webhook** to: `https://<user>--hermes-api.modal.run/twilio/voice`.
3.  The system will automatically generate the TwiML needed to start the bidirectional `<Connect><Stream>` WebSocket.

---
**Status:** Deploy-Verified
