"""Global test configuration and fixtures."""

import sys
from unittest.mock import MagicMock

# 1. Mock chatterbox and its transitive dependencies that cause issues
mock_chatterbox = MagicMock()
mock_chatterbox_tts = MagicMock()
mock_chatterbox.tts = mock_chatterbox_tts

sys.modules["chatterbox"] = mock_chatterbox
sys.modules["chatterbox.tts"] = mock_chatterbox_tts
sys.modules["perth"] = MagicMock()
sys.modules["librosa"] = MagicMock()
sys.modules["onnx"] = MagicMock()
sys.modules["ml_dtypes"] = MagicMock()
sys.modules["s3tokenizer"] = MagicMock()
