# API Reference

Complete reference for Hermes API endpoints.

## Base URL

```
Development: http://localhost:8000
Production: https://api.hermes.example.com
```

## Authentication

Hermes uses Twilio request validation for WebSocket connections. HTTP endpoints don't require authentication in the current implementation (add as needed).

## Endpoints

### Health Check

Check application health status.

```http
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "environment": "production"
}
```

### Readiness Check

Check if application is ready to serve requests.

```http
GET /ready
```

**Response:**

```json
{
  "status": "ready",
  "checks": {
    "database": true,
    "redis": true,
    "vector_db": true
  }
}
```

### Liveness Check

Simple liveness probe.

```http
GET /live
```

**Response:**

```json
{
  "status": "alive",
  "version": "0.1.0",
  "environment": "production"
}
```

### Metrics

Prometheus metrics endpoint.

```http
GET /metrics
```

**Response:**

Prometheus-formatted metrics including:
- `hermes_active_calls`: Gauge of active calls
- `hermes_calls_total`: Counter of calls by status
- `hermes_stt_latency_seconds`: STT latency histogram
- `hermes_llm_latency_seconds`: LLM latency histogram
- `hermes_tts_latency_seconds`: TTS latency histogram

### WebSocket Stream

Primary WebSocket endpoint for Twilio media streams.

```http
WS /stream/{call_sid}
```

**Path Parameters:**

| Name | Type | Description |
|------|------|-------------|
| call_sid | string | Twilio Call SID |

**Message Format:**

Incoming messages from Twilio:

#### Connected

```json
{
  "event": "connected",
  "protocol": "call",
  "version": "1.0"
}
```

#### Start

```json
{
  "event": "start",
  "sequenceNumber": 1,
  "start": {
    "callSid": "CA1234567890abcdef",
    "accountSid": "AC1234567890abcdef",
    "streamSid": "MZ1234567890abcdef"
  }
}
```

#### Media

```json
{
  "event": "media",
  "sequenceNumber": 2,
  "streamSid": "MZ1234567890abcdef",
  "media": {
    "track": "inbound",
    "chunk": "1",
    "timestamp": "1234567890",
    "payload": "base64-encoded-audio"
  }
}
```

Audio payload is mu-law encoded, 8kHz, mono.

#### DTMF

```json
{
  "event": "dtmf",
  "streamSid": "MZ1234567890abcdef",
  "sequenceNumber": 3,
  "dtmf": {
    "digit": "1",
    "track": "inbound"
  }
}
```

#### Stop

```json
{
  "event": "stop",
  "sequenceNumber": 4,
  "streamSid": "MZ1234567890abcdef",
  "stop": {
    "callSid": "CA1234567890abcdef",
    "accountSid": "AC1234567890abcdef"
  }
}
```

**Outgoing Messages:**

#### Media Response

Send audio back to Twilio:

```json
{
  "event": "media",
  "streamSid": "MZ1234567890abcdef",
  "media": {
    "payload": "base64-encoded-audio"
  }
}
```

#### Mark

Send timing mark:

```json
{
  "event": "mark",
  "streamSid": "MZ1234567890abcdef",
  "mark": {
    "name": "response-complete"
  }
}
```

#### Clear

Clear audio buffer:

```json
{
  "event": "clear",
  "streamSid": "MZ1234567890abcdef"
}
```

### Test Client (Development)

Simple WebSocket test client page.

```http
GET /stream/test-client
```

Returns an HTML page with a basic WebSocket test client.

## Error Handling

### HTTP Errors

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request |
| 404 | Not Found |
| 500 | Internal Server Error |

**Error Response Format:**

```json
{
  "detail": "Error description"
}
```

### WebSocket Errors

WebSocket connections may close with specific codes:

| Code | Description |
|------|-------------|
| 4001 | Call SID mismatch |
| 4002 | Service unavailable |
| 4003 | Rate limited |
| 1011 | Internal server error |

## Rate Limiting

Rate limits can be configured per environment:

- Development: No limits
- Staging: 100 requests/minute
- Production: 1000 requests/minute

## SDK Examples

### Python Client

```python
import asyncio
import json
import websockets

async def call_hermes():
    call_sid = "test-call-123"
    uri = f"ws://localhost:8000/stream/{call_sid}"

    async with websockets.connect(uri) as websocket:
        # Send start
        await websocket.send(json.dumps({
            "event": "start",
            "sequenceNumber": 1,
            "start": {
                "callSid": call_sid,
                "accountSid": "test",
                "streamSid": "test-stream"
            }
        }))

        # Send audio (mu-law encoded silence)
        import base64
        audio = base64.b64encode(b"\xff" * 160).decode()

        await websocket.send(json.dumps({
            "event": "media",
            "sequenceNumber": 2,
            "streamSid": "test-stream",
            "media": {
                "track": "inbound",
                "chunk": "1",
                "timestamp": "0",
                "payload": audio
            }
        }))

        # Receive response
        response = await websocket.recv()
        print(f"Received: {response}")

asyncio.run(call_hermes())
```

### JavaScript/Node.js

```javascript
const WebSocket = require('ws');

const callSid = 'test-call-123';
const ws = new WebSocket(`ws://localhost:8000/stream/${callSid}`);

ws.on('open', () => {
    // Send start message
    ws.send(JSON.stringify({
        event: 'start',
        sequenceNumber: 1,
        start: {
            callSid: callSid,
            accountSid: 'test',
            streamSid: 'test-stream'
        }
    }));
});

ws.on('message', (data) => {
    console.log('Received:', data.toString());
});

ws.on('close', () => {
    console.log('Connection closed');
});
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Metrics
curl http://localhost:8000/metrics

# WebSocket (requires a WebSocket client like wscat)
npm install -g wscat
wscat -c ws://localhost:8000/stream/test-call
```

## Webhook Integration

### Twilio Integration

Configure Twilio to send calls to Hermes:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Start>
        <Stream url="wss://your-hermes-server.com/stream/{CallSid}"/>
    </Start>
    <Say>Connecting you to our AI assistant.</Say>
    <Pause length="60"/>
</Response>
```

Or using TwiML Bin:

```
{
  "url": "wss://your-hermes-server.com/stream/{{CallSid}}",
  "track": "both_tracks"
}
```

## OpenAPI/Swagger

Interactive API documentation is available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

## Versioning

API versioning follows semantic versioning:

- Current version: 0.1.0
- Breaking changes bump major version
- New features bump minor version
- Bug fixes bump patch version

Version is included in:
- HTTP headers: `X-API-Version: 0.1.0`
- Health check response
