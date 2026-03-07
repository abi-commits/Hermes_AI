"""Smoke test for the Hermes remote Modal TTS path."""

from __future__ import annotations

import argparse
import asyncio

from config import get_settings
from hermes.services.tts import ModalRemoteTTSService


async def _run_smoke_test(text: str, min_stream_chunks: int) -> int:
    """Validate that the remote Modal TTS worker returns PCM bytes Hermes can use."""
    settings = get_settings()

    service = ModalRemoteTTSService(
        app_name=settings.modal_tts_app_name,
        class_name=settings.modal_tts_class_name,
        sample_rate=settings.modal_tts_sample_rate,
        default_chunk_size=settings.modal_tts_chunk_size,
        default_embed_watermark=settings.modal_tts_embed_watermark,
    )

    print("Remote TTS target:")
    print(f"  app={settings.modal_tts_app_name}")
    print(f"  class={settings.modal_tts_class_name}")
    print(f"  sample_rate={service.sample_rate}")

    full_audio = await service.generate(text)
    if not full_audio:
        raise RuntimeError("Remote generate() returned empty audio")
    if len(full_audio) % 2 != 0:
        raise RuntimeError("Remote generate() did not return int16 PCM bytes")

    print("Full generation:")
    print(f"  bytes={len(full_audio)}")
    print(f"  samples={len(full_audio) // 2}")

    stream_chunks = 0
    stream_bytes = 0
    async for chunk in service.generate_stream(text):
        if not chunk:
            continue
        if len(chunk) % 2 != 0:
            raise RuntimeError("Remote stream chunk did not return int16 PCM bytes")
        stream_chunks += 1
        stream_bytes += len(chunk)

    if stream_chunks < min_stream_chunks:
        raise RuntimeError(
            f"Remote stream returned too few chunks: {stream_chunks} < {min_stream_chunks}"
        )

    print("Streaming generation:")
    print(f"  chunks={stream_chunks}")
    print(f"  bytes={stream_bytes}")

    print("Remote Modal TTS smoke test passed.")
    return 0


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Smoke test the remote Modal TTS adapter")
    parser.add_argument(
        "--text",
        default="Hermes remote text to speech smoke test.",
        help="Text to synthesize during the smoke test.",
    )
    parser.add_argument(
        "--min-stream-chunks",
        type=int,
        default=1,
        help="Minimum number of streamed chunks required for success.",
    )
    args = parser.parse_args()

    get_settings.cache_clear()
    return asyncio.run(_run_smoke_test(args.text, args.min_stream_chunks))


if __name__ == "__main__":
    raise SystemExit(main())
