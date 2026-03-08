# Hermes Voice Support Service

A production-grade AI-powered voice support service built with FastAPI, WebSockets, and modern ML services.

## Overview

Hermes provides real-time voice assistance by integrating:
- **Speech-to-Text (STT)**: Deepgram for accurate transcription
- **Large Language Model (LLM)**: Gemini for intelligent responses
- **Text-to-Speech (TTS)**: Chatterbox for natural voice synthesis
- **RAG**: Vector database for knowledge retrieval

## Architecture

### Current Production Deployment

```text
┌────────────────────┐
│       Caller       │
│   PSTN / Mobile    │
└─────────┬──────────┘
          │ phone call
          ▼
┌────────────────────┐
│       Twilio       │
│  Voice + Media WS  │
└───────┬─────┬──────┘
        │     │
        │     └─────────────────────────────── outbound audio (8 kHz mu-law)
        │
        │ webhook + bidirectional WebSocket
        ▼
┌────────────────────────────────────────────────────────────────────┐
│                    Modal: Hermes API Service                      │
│                                                                    │
│  FastAPI app                                                       │
│  - /twilio/voice                                                   │
│  - /twilio/status                                                  │
│  - /stream/{call_sid}                                              │
│                                                                    │
│  Runtime components                                                │
│  - ConnectionManager                                               │
│  - CallOrchestrator                                                │
│  - Per-call state machine                                          │
│  - Audio resample + mu-law encoding for Twilio                     │
│                                                                    │
│  Service clients                                                   │
│  - Deepgram STT                                                    │
│  - Gemini LLM                                                      │
│  - Chroma RAG                                                      │
│  - ModalRemoteTTSService                                           │
└───────────────┬───────────────────────┬───────────────────────┬────┘
                │                       │                       │
                │ transcripts           │ prompts/context       │ retrieval
                ▼                       ▼                       ▼
       ┌────────────────┐      ┌────────────────┐     ┌──────────────────┐
       │    Deepgram    │      │     Gemini     │     │ Chroma Cloud /   │
       │      STT       │      │      LLM       │     │ External VectorDB│
       └────────────────┘      └────────────────┘     └──────────────────┘
                │
                │ PCM stream requests
                ▼
┌────────────────────────────────────────────────────────────────────┐
│                Modal: Dedicated GPU TTS Worker                    │
│                                                                    │
│  RemoteChatterboxTTSWorker                                         │
│  - GPU-backed Chatterbox model                                     │
│  - returns native-rate PCM                                         │
│  - streams PCM chunks back to Hermes                               │
│                                                                    │
│  Default GPU: L4                                                   │
└────────────────────────────────────────────────────────────────────┘
```

### Responsibilities

- Twilio: telephony, inbound call webhook, bidirectional media stream
- Hermes API on Modal: call control, orchestration, STT/LLM/RAG coordination, Twilio-compatible audio output
- Dedicated Modal GPU worker: text-to-speech synthesis only
- Deepgram: speech-to-text
- Gemini: response generation
- Chroma: retrieval-augmented context

### Why the TTS worker is separate

The API service keeps ownership of:

- live call state
- interruption and barge-in behavior
- Twilio audio formatting
- orchestration across STT, LLM, and RAG

The GPU worker only handles:

- Chatterbox inference
- returning native PCM audio back to Hermes

## Quick Start

### Prerequisites

- Python 3.11+
- uv (modern Python package manager)
- Docker & Docker Compose (optional)
- Twilio account
- Deepgram API key
- Gemini API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-org/hermes.git
cd hermes
```

2. Install dependencies:
```bash
uv sync --extra all
```

3. Copy environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. Run with Docker Compose:
```bash
docker-compose up -d
```

Or run locally:
```bash
uv run uvicorn hermes.main:app --reload
```

## Development

### Project Structure

```
hermes/
├── config/                 # Configuration management
├── hermes/                 # Main application
│   ├── websocket/          # WebSocket handling
│   ├── core/              # Domain logic
│   ├── services/          # External integrations
│   ├── workers/           # Background tasks
│   ├── models/            # Data models
│   ├── api/               # HTTP endpoints
│   └── utils/             # Utilities
├── tests/                  # Test suite
├── scripts/               # Helper scripts
└── docs/                  # Documentation
```

### Common Commands

```bash
# Run tests
make test

# Run linting
make lint

# Format code
make format

# Type checking
make type-check

# Run locally
make run

# Build Docker image
make build
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=hermes --cov-report=html

# Run specific test category
pytest -m unit
pytest -m integration
pytest -m "not slow"
```

## Deployment

### Docker

```bash
docker build -t hermes .
docker run -p 8000:8000 --env-file .env hermes
```

### Kubernetes

See `docs/deployment.md` for Kubernetes deployment instructions.

## API Documentation

Once running, API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Monitoring

Prometheus metrics are exposed at `/metrics`:
- Active calls
- STT/TTS latency
- LLM token usage
- Error rates

## Contributing

1. Create a feature branch
2. Make your changes
3. Run tests and linting
4. Submit a pull request

## License

MIT License - see LICENSE file for details.
