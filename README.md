# Hermes AI

Hermes AI is a high-performance, low-latency voice AI orchestration platform designed for real-time telephony. It transforms raw audio streams into natural, context-aware voice conversations using an advanced Ear-Brain-Mouth pipeline.

## 🚀 Core Features

*   **Zero-Latency Greetings:** Immediate AI response upon connection, bypassing LLM "thinking" time.
*   **Agentic RAG:** Intelligent information retrieval with ChromaDB and filler speech to mask processing latency.
*   **Barge-In Support:** Seamless conversation flow with real-time speech detection and audio interruption.
*   **Modal Infrastructure:** Serverless GPU workers for high-speed speech synthesis (TTS) and scalable API hosting.
*   **Production Observability:** Unified structured JSON logging and real-time "Time to First Byte" (TTFB) telemetry.

## 🏗️ System Architecture

Hermes operates as a parallel relay race:
1.  **Ears (STT):** Deepgram converts 8kHz mu-law telephony audio into text.
2.  **Brain (LLM):** Google Gemini 2.5 Flash processes text, retrieves data, and streams sentences.
3.  **Mouth (TTS):** Custom Chatterbox model on NVIDIA T4 GPUs synthesizes high-quality audio chunks.

## 📁 Project Structure

```text
├── hermes/               # Core application logic
│   ├── core/             # Orchestrator and Call state machines
│   ├── services/         # STT, LLM, TTS, and RAG implementations
│   ├── api/              # REST endpoints (Health, Ready, Twilio)
│   └── websocket/        # Real-time media stream handling
├── modal_deploy/         # Infrastructure-as-Code for Modal
├── scripts/              # Diagnostic and seeding tools
└── docs/                 # Detailed technical documentation
```

## 🚦 Quick Start

Detailed instructions for setup and deployment can be found in the `docs/` directory:

1.  **[Architecture Guide](docs/ARCHITECTURE.md):** How the conversation engine works.
2.  **[Deployment Guide](docs/DEPLOYMENT.md):** Moving from local to Modal Production.
3.  **[Development Guide](docs/DEVELOPMENT.md):** Local testing and diagnostics.

---
**Status:** Production-Ready | Logic-Verified
