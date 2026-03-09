"""Test Modal TTS service directly without WebSocket."""

import asyncio
import os
from pathlib import Path

# Change to project root
os.chdir(Path(__file__).parents[1])

import structlog

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
)
logger = structlog.get_logger(__name__)


async def test_modal_tts_ping():
    """Test if Modal TTS worker is accessible via ping."""
    from hermes.services.tts import ModalRemoteTTSService

    logger.info("=" * 60)
    logger.info("TEST 1: Modal TTS Ping Test")
    logger.info("=" * 60)

    tts = ModalRemoteTTSService(
        app_name="hermes-tts",
        class_name="RemoteChatterboxTTSWorker",
        sample_rate=24_000,
    )

    try:
        result = await tts.ping()
        logger.info(f"Ping result: {result}")
        return result
    except Exception as e:
        logger.error(f"Ping failed: {e}", exc_info=True)
        return False


async def test_modal_tts_generate_full():
    """Test full audio generation (not streaming)."""
    from hermes.services.tts import ModalRemoteTTSService

    logger.info("=" * 60)
    logger.info("TEST 2: Modal TTS Full Generate Test")
    logger.info("=" * 60)

    tts = ModalRemoteTTSService(
        app_name="hermes-tts",
        class_name="RemoteChatterboxTTSWorker",
        sample_rate=24_000,
    )

    test_text = "Hello, this is a test."

    try:
        logger.info(f"Generating audio for: '{test_text}'")
        audio = await tts.generate(test_text)
        logger.info(f"Generated {len(audio)} bytes of audio")

        if len(audio) > 0:
            logger.info("SUCCESS: Full generation works")
            return True
        else:
            logger.error("ERROR: Empty audio received")
            return False

    except Exception as e:
        logger.error(f"Generate failed: {e}", exc_info=True)
        return False


async def test_modal_tts_generate_stream():
    """Test streaming audio generation."""
    from hermes.services.tts import ModalRemoteTTSService

    logger.info("=" * 60)
    logger.info("TEST 3: Modal TTS Streaming Generate Test")
    logger.info("=" * 60)

    tts = ModalRemoteTTSService(
        app_name="hermes-tts",
        class_name="RemoteChatterboxTTSWorker",
        sample_rate=24_000,
        default_chunk_size=50,
    )

    test_text = "Hello, this is a streaming test."

    try:
        logger.info(f"Streaming audio for: '{test_text}'")
        chunk_count = 0
        total_bytes = 0

        async for chunk in tts.generate_stream(test_text):
            chunk_count += 1
            total_bytes += len(chunk)
            if chunk_count <= 3:
                logger.info(f"Chunk {chunk_count}: {len(chunk)} bytes")

        logger.info(f"Stream complete: {chunk_count} chunks, {total_bytes} total bytes")

        if chunk_count > 0:
            logger.info("SUCCESS: Streaming generation works")
            return True
        else:
            logger.error("ERROR: No chunks received from stream")
            return False

    except Exception as e:
        logger.error(f"Stream failed: {e}", exc_info=True)
        return False


async def test_modal_tts_sample_rate():
    """Test if sample rate matches between client and worker."""
    from hermes.services.tts import ModalRemoteTTSService

    logger.info("=" * 60)
    logger.info("TEST 4: Modal TTS Sample Rate Check")
    logger.info("=" * 60)

    tts = ModalRemoteTTSService(
        app_name="hermes-tts",
        class_name="RemoteChatterboxTTSWorker",
        sample_rate=24_000,
    )

    logger.info(f"Client-side sample rate: {tts.sample_rate}")

    try:
        instance = await tts._get_remote_instance()
        remote_sr = await tts._call_remote(instance.get_sample_rate)
        logger.info(f"Remote worker sample rate: {remote_sr}")

        if tts.sample_rate == remote_sr:
            logger.info("SUCCESS: Sample rates match")
            return True
        else:
            logger.error(f"MISMATCH: Client {tts.sample_rate} vs Remote {remote_sr}")
            return False

    except Exception as e:
        logger.error(f"Sample rate check failed: {e}", exc_info=True)
        return False


async def main():
    logger.info("Starting Modal TTS Direct Tests")
    logger.info("These tests require the Modal TTS worker to be deployed")
    logger.info("")

    results = []

    # Test 1: Ping
    results.append(("Ping", await test_modal_tts_ping()))

    # Test 2: Full generate (only if ping succeeded)
    if results[0][1]:
        results.append(("Full Generate", await test_modal_tts_generate_full()))
    else:
        logger.warning("Skipping full generate test - ping failed")
        results.append(("Full Generate", None))

    # Test 3: Streaming (only if ping succeeded)
    if results[0][1]:
        results.append(("Stream Generate", await test_modal_tts_generate_stream()))
    else:
        logger.warning("Skipping stream test - ping failed")
        results.append(("Stream Generate", None))

    # Test 4: Sample rate (only if ping succeeded)
    if results[0][1]:
        results.append(("Sample Rate Check", await test_modal_tts_sample_rate()))
    else:
        logger.warning("Skipping sample rate test - ping failed")
        results.append(("Sample Rate Check", None))

    # Summary
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    for name, result in results:
        status = "PASS" if result else ("FAIL" if result is False else "SKIP")
        logger.info(f"  {name}: {status}")

    # Analysis
    ping_ok = results[0][1]
    generate_ok = results[1][1] if len(results) > 1 else None
    stream_ok = results[2][1] if len(results) > 2 else None

    if not ping_ok:
        logger.error("\nCONCLUSION: Modal TTS worker is not accessible.")
        logger.error("  - Check if Modal app 'hermes-tts' is deployed: modal deploy modal_deploy/tts.py")
        logger.error("  - Verify class name 'RemoteChatterboxTTSWorker'")
        logger.error("  - Check Modal dashboard for errors")
    elif ping_ok and not generate_ok:
        logger.error("\nCONCLUSION: Worker is reachable but generation fails.")
        logger.error("  - Check Modal worker logs for errors")
        logger.error("  - Verify the worker has GPU resources")
    elif ping_ok and generate_ok and not stream_ok:
        logger.error("\nCONCLUSION: Full generation works but streaming fails.")
        logger.error("  - This suggests an issue with Modal's generator handling")
        logger.error("  - Check if the Modal SDK versions match between client and worker")
    elif all(r[1] for r in results if r[1] is not None):
        logger.info("\nCONCLUSION: All Modal TTS tests passed!")
        logger.info("  - The issue is likely in the WebSocket/greeting flow")
        logger.info("  - Check the server-side logs for greeting handling")


if __name__ == "__main__":
    asyncio.run(main())
