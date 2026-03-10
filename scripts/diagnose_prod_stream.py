import asyncio
import json
import websockets
import time
import base64
import sys

async def diagnose_stream(local=False, prompt=None):
    call_sid = "diag_test_local" if local else "diag_test_prod"
    if local:
        uri = f"ws://localhost:8000/stream/{call_sid}"
    else:
        uri = f"wss://abinesh3200--hermes-api.modal.run/stream/{call_sid}"
        
    print(f"Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri, open_timeout=30.0) as websocket:
            print("✅ Socket connected.")
            
            # 1. Twilio Handshake Step 1: connected
            print("📤 Sending 'connected' event...")
            await websocket.send(json.dumps({
                "event": "connected",
                "protocol": "Call",
                "version": "1.0.0"
            }))
            
            # 2. Twilio Handshake Step 2: start
            print("📤 Sending 'start' event...")
            handshake = {
                "event": "start",
                "sequenceNumber": "1",
                "start": {
                    "accountSid": "AC_LOCAL_TEST",
                    "callSid": call_sid,
                    "streamSid": "MZ_LOCAL_TEST",
                    "tracks": ["inbound"],
                    "customParameters": {
                        "greeting": "Congratulations! Production system is active."
                    }
                }
            }
            # ── NEW: Inject test prompt into customParameters ──
            if prompt:
                handshake["start"]["customParameters"]["test_prompt"] = prompt
                print(f"📝 Injected Test Prompt: '{prompt}'")

            await websocket.send(json.dumps(handshake))

            # 3. Priming: Send enough audio to satisfy Deepgram's initial buffers
            print("📤 Sending priming audio (400ms of valid mu-law silence)...")
            # 0x7f is standard mu-law silence
            silence_chunk = base64.b64encode(b"\x7f" * 160).decode()
            for i in range(20): # Send 400ms of audio (20 * 20ms)
                media_msg = {
                    "event": "media",
                    "sequenceNumber": str(i + 2),
                    "streamSid": "MZ_LOCAL_TEST",
                    "media": {
                        "track": "inbound",
                        "chunk": str(i),
                        "timestamp": str(int(time.time() * 1000)),
                        "payload": silence_chunk
                    }
                }
                await websocket.send(json.dumps(media_msg))
                await asyncio.sleep(0.01) # Burst mode

            print("⏳ Waiting for server audio chunks...")
            
            start_time = time.perf_counter()
            chunks_received = 0
            first_byte_time = None
            
            # Use a global deadline instead of per-message timeout
            wait_limit = 60.0 if local else 120.0
            global_deadline = time.perf_counter() + wait_limit
            
            while chunks_received < 100: # Increase limit for LLM response
                remaining = global_deadline - time.perf_counter()
                if remaining <= 0:
                    print("⌛ Global timeout reached.")
                    break
                    
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=min(remaining, 2.0))
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
                    # Inner timeout just to loop back and check global deadline
                    continue
                except Exception as e:
                    print(f"❌ Error receiving: {e}")
                    break
            
            # 4. Clean Teardown
            print("📤 Sending 'stop' event for clean teardown...")
            await websocket.send(json.dumps({
                "event": "stop",
                "sequenceNumber": "99",
                "streamSid": "MZ_LOCAL_TEST",
                "stop": {"accountSid": "AC_LOCAL_TEST", "callSid": call_sid}
            }))

            print("\n--- Diagnostic Results ---")
            print(f"Total Chunks: {chunks_received}")
            if first_byte_time:
                print("✅ STREAM VERIFIED.")
            else:
                print("❌ NO AUDIO RECEIVED.")
                
    except Exception as e:
        print(f"❌ Connection Failed: {e}")

if __name__ == "__main__":
    is_local = "--local" in sys.argv
    prompt = None
    for i, arg in enumerate(sys.argv):
        if arg == "--prompt" and i + 1 < len(sys.argv):
            prompt = sys.argv[i+1]
            break
            
    asyncio.run(diagnose_stream(local=is_local, prompt=prompt))
