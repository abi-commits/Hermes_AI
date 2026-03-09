import asyncio
import json
import websockets
import time
import base64

async def diagnose_production_stream():
    uri = "wss://abinesh3200--hermes-api.modal.run/stream/diag_test_active"
    print(f"Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri, open_timeout=30.0) as websocket:
            print("✅ Handshake successful. Sending Start Event...")
            
            handshake = {
                "event": "start",
                "sequenceNumber": "1",
                "start": {
                    "accountSid": "AC_PROD_TEST",
                    "callSid": "diag_test_active",
                    "streamSid": "MZ_PROD_TEST",
                    "tracks": ["inbound"],
                    "customParameters": {
                        "greeting": "System ready. Initializing active stream verification."
                    }
                }
            }
            await websocket.send(json.dumps(handshake))

            # ── NEW: Send dummy audio chunks to prime the system ──
            # Sending 5 chunks of 160 bytes of silence (mu-law)
            print("📤 Sending dummy audio chunks to prime STT/Orchestrator...")
            silence_chunk = base64.b64encode(b"\xff" * 160).decode()
            for i in range(5):
                media_msg = {
                    "event": "media",
                    "sequenceNumber": str(i + 2),
                    "streamSid": "MZ_PROD_TEST",
                    "media": {
                        "track": "inbound",
                        "chunk": str(i),
                        "timestamp": str(int(time.time() * 1000)),
                        "payload": silence_chunk
                    }
                }
                await websocket.send(json.dumps(media_msg))
                await asyncio.sleep(0.02) # Simulate 20ms packets

            print("⏳ Waiting for audio chunks (Timeout: 120s)...")
            
            start_time = time.perf_counter()
            chunks_received = 0
            first_byte_time = None
            
            while chunks_received < 50:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=120.0)
                    data = json.loads(message)
                    
                    event = data.get("event")
                    
                    if event == "media":
                        if chunks_received == 0:
                            first_byte_time = time.perf_counter() - start_time
                            print(f"🚀 SUCCESS: First chunk received in {first_byte_time:.2f}s")
                        
                        chunks_received += 1
                        if chunks_received % 10 == 0:
                            print(f"🔊 Received {chunks_received} audio chunks...")
                            
                    elif event == "mark":
                        print(f"📍 Received Marker: {data.get('mark', {}).get('name')}")
                        
                except asyncio.TimeoutError:
                    print("⌛ Timeout reached waiting for audio.")
                    break
            
            print("\n--- Diagnostic Results ---")
            print(f"Total Chunks: {chunks_received}")
            if first_byte_time:
                print("✅ END-TO-END STREAM VERIFIED.")
            else:
                print("❌ NO AUDIO RECEIVED.")
                
    except Exception as e:
        print(f"❌ Connection Failed: {e}")

if __name__ == "__main__":
    asyncio.run(diagnose_production_stream())
