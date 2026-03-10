import asyncio
import os
import modal
from dotenv import load_dotenv

async def test_modal_tts_direct():
    load_dotenv()
    
    app_name = "hermes-tts"
    class_name = "RemoteChatterboxTTSWorker"
    
    print(f"Connecting to Modal App: {app_name}, Class: {class_name}...")
    
    try:
        # Resolve class
        RemoteCls = modal.Cls.from_name(app_name, class_name)
        worker = RemoteCls()
        
        print("Calling generate_stream.remote_gen...")
        text = "This is a direct test of the Modal TTS worker. If you hear this, the worker is functional."
        
        chunk_count = 0
        total_bytes = 0
        start_time = asyncio.get_event_loop().time()
        
        # Test direct remote_gen
        async for chunk in worker.generate_stream.remote_gen.aio(text=text):
            if chunk_count == 0:
                ttfb = (asyncio.get_event_loop().time() - start_time) * 1000
                print(f"🚀 TTFB: {ttfb:.2f}ms")
            
            chunk_count += 1
            total_bytes += len(chunk)
            if chunk_count % 10 == 0:
                print(f"🔊 Received {chunk_count} chunks...")

        print("\n--- Results ---")
        print(f"Total Chunks: {chunk_count}")
        print(f"Total Bytes: {total_bytes}")
        if chunk_count > 0:
            print("✅ MODAL WORKER IS FUNCTIONAL.")
        else:
            print("❌ MODAL WORKER RETURNED NO DATA.")
            
    except Exception as e:
        print(f"💥 Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_modal_tts_direct())
