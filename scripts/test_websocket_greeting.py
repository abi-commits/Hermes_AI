"""Quick focused test of WebSocket greeting flow."""

import asyncio
import json
import websockets
import time

async def test_greeting():
    uri = "wss://abinesh3200--hermes-api.modal.run/stream/test_greeting_001"
    print(f"Connecting to {uri}...")

    async with websockets.connect(uri, open_timeout=30.0) as ws:
        print("✅ Connected")

        # Send connected event
        await ws.send(json.dumps({
            "event": "connected",
            "protocol": "Call",
            "version": "1.0.0"
        }))
        print("📤 Sent 'connected' event")

        # Send start event with greeting
        greeting_text = "Hello! This is a greeting message from the test."
        await ws.send(json.dumps({
            "event": "start",
            "sequenceNumber": "1",
            "start": {
                "accountSid": "AC_TEST",
                "callSid": "test_greeting_001",
                "streamSid": "STREAM_TEST_001",
                "tracks": ["inbound"],
                "customParameters": {
                    "greeting": greeting_text
                }
            }
        }))
        print(f"📤 Sent 'start' event with greeting: '{greeting_text}'")

        # Wait for events with shorter timeout
        print("⏳ Waiting for server responses (10s timeout)...")
        start_time = time.perf_counter()

        while time.perf_counter() - start_time < 10:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                data = json.loads(msg)
                event = data.get("event")

                if event == "media":
                    payload = data.get("media", {}).get("payload", "")
                    print(f"🎵 Received media: {len(payload)} bytes base64 payload")
                elif event == "mark":
                    print(f"📍 Mark: {data.get('mark', {}).get('name')}")
                elif event == "clear":
                    print("🧹 Clear event")
                else:
                    print(f"📨 Other event: {event}, data: {json.dumps(data)[:100]}")

            except asyncio.TimeoutError:
                print("⏱️ No message in 2s...")
                continue

        print("\n--- Test Complete ---")
        print("If no media events were received, check Modal dashboard logs for:")
        print("  - start_event_greeting")
        print("  - manager_connect_starting")
        print("  - tts_task_processing_text")
        print("  - calling_remote_generator")

if __name__ == "__main__":
    asyncio.run(test_greeting())
