import asyncio
import os
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)
from dotenv import load_dotenv

async def test_deepgram():
    load_dotenv()
    api_key = os.getenv("DEEPGRAM_API_KEY")
    print(f"Testing with API Key: {api_key[:4]}...{api_key[-4:]}")
    
    try:
        config = DeepgramClientOptions(options={"keepalive": "true"})
        client = DeepgramClient(api_key, config)
        
        options = LiveOptions(
            model="nova-2",
            language="en-US",
            smart_format=True,
            interim_results=True,
            encoding="mulaw",
            sample_rate=8000,
            channels=1,
            utterance_end_ms=700, # Testing this now
        )
        
        connection = client.listen.asyncwebsocket.v("1")
        
        async def on_open(self, open, **kwargs):
            print("✅ Connection Open")
            
        async def on_error(self, error, **kwargs):
            print(f"❌ Connection Error: {error}")

        connection.on(LiveTranscriptionEvents.Open, on_open)
        connection.on(LiveTranscriptionEvents.Error, on_error)
        
        print("Connecting with utterance_end_ms...")
        if await connection.start(options):
            print("🎉 SUCCESS: Connected to Deepgram with utterance_end_ms!")
            await connection.finish()
        else:
            print("❌ FAILED: Could not start connection.")
            
    except Exception as e:
        print(f"💥 Exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_deepgram())
