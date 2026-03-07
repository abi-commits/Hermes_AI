"""Script to verify TTS streaming functionality."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import mocks from conftest to ensure environment is clean
import tests.conftest 

from hermes.services.tts.chatterbox import ChatterboxTTSService
from hermes.services.tts.mock import MockTTSService


async def test_mock_streaming():
    """Test streaming with MockTTSService."""
    print("\n--- Testing MockTTSService Streaming ---")
    service = MockTTSService(duration_seconds=0.5, chunk_size=1000)
    
    chunks_received = 0
    total_bytes = 0
    
    async for chunk in service.generate_stream("Hello, this is a mock test."):
        chunks_received += 1
        total_bytes += len(chunk)
        print(f"Received chunk {chunks_received}: {len(chunk)} bytes")
        
    print(f"Total chunks: {chunks_received}, Total bytes: {total_bytes}")
    
    # 0.5s * 16000Hz * 2 bytes = 16000 bytes
    expected_bytes = 16000
    if total_bytes == expected_bytes:
        print("✅ Mock streaming: Correct number of bytes generated.")
    else:
        print(f"❌ Mock streaming: Expected {expected_bytes} bytes, got {total_bytes}.")


async def test_chatterbox_streaming_mocked_model():
    """Test ChatterboxTTSService with a mocked model."""
    print("\n--- Testing ChatterboxTTSService (Mocked Model) ---")
    
    # We patch ChatterboxTTS which is now a MagicMock from conftest
    with patch("chatterbox.tts.ChatterboxTTS") as MockChatterboxTTS:
        mock_model = MagicMock()
        mock_model.sr = 24000
        
        # Mock generate_stream to yield some fake chunks
        fake_wav = torch.randn(1, 2400)
        mock_model.generate_stream.return_value = [(fake_wav, {"latency": 0.01})]
        MockChatterboxTTS.from_pretrained.return_value = mock_model
        
        service = ChatterboxTTSService(device="cpu")
        
        chunks_received = 0
        total_bytes = 0
        
        # We use a small chunk size to see it yielding
        async for chunk in service.generate_stream("Hello test", chunk_size=1000):
            chunks_received += 1
            total_bytes += len(chunk)
            print(f"Received chunk {chunks_received}: {len(chunk)} bytes")
            
        print(f"Total chunks: {chunks_received}, Total bytes: {total_bytes}")
        
        if total_bytes > 0:
            print("✅ Chatterbox streaming: Audio bytes generated successfully.")
        else:
            print("❌ Chatterbox streaming: No audio bytes generated.")


async def main():
    """Main entry point."""
    await test_mock_streaming()
    try:
        await test_chatterbox_streaming_mocked_model()
    except Exception as e:
        print(f"❌ Chatterbox streaming test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
