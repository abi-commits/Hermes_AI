"""Benchmark TTS latency.

Usage:
    python scripts/benchmark_tts.py [options]

Examples:
    python scripts/benchmark_tts.py
    python scripts/benchmark_tts.py --samples 100 --text-lengths 50 100 200
"""

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import structlog

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes.services.tts import ChatterboxTTSService

logger = structlog.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark TTS latency")
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples per text length (default: 50)",
    )
    parser.add_argument(
        "--text-lengths",
        nargs="+",
        type=int,
        default=[50, 100, 200, 500],
        help="Text lengths to benchmark (in characters, default: 50 100 200 500)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup samples (default: 5)",
    )
    parser.add_argument(
        "--provider",
        choices=["chatterbox", "openai", "mock"],
        default="chatterbox",
        help="TTS provider to benchmark",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON)",
    )

    return parser.parse_args()


def generate_test_text(length: int) -> str:
    """Generate test text of specified length.

    Args:
        length: Desired text length.

    Returns:
        Test text.
    """
    sentences = [
        "Hello, this is a test sentence for TTS benchmarking.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming how we interact with technology.",
        "Voice assistants provide a natural way to access information and services.",
        "Real-time text-to-speech systems require low latency for good user experience.",
        "Machine learning models continue to improve in both quality and speed.",
        "Cloud-based services offer scalable solutions for voice applications.",
        "The integration of speech recognition and synthesis creates seamless interactions.",
        "Users expect fast and accurate responses from voice-enabled systems.",
        "Testing different text lengths helps understand performance characteristics.",
    ]

    text = ""
    while len(text) < length:
        for sentence in sentences:
            if len(text) + len(sentence) + 1 <= length:
                text += sentence + " "
            else:
                break

    return text[:length].strip()


async def benchmark_single(text: str, tts_service: ChatterboxTTSService) -> dict[str, Any]:
    """Benchmark a single TTS synthesis.

    Args:
        text: Text to synthesize.
        tts_service: TTS service instance.

    Returns:
        Benchmark results.
    """
    start_time = time.perf_counter()

    try:
        audio_bytes = await tts_service.generate(text)
        end_time = time.perf_counter()

        latency = end_time - start_time
        # audio_bytes are int16: 2 bytes per sample
        num_samples = len(audio_bytes) // 2
        audio_duration = num_samples / tts_service.sample_rate

        return {
            "success": True,
            "latency": latency,
            "text_length": len(text),
            "audio_samples": num_samples,
            "audio_duration": audio_duration,
            "rtf": latency / audio_duration if audio_duration > 0 else 0,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "latency": time.perf_counter() - start_time,
            "text_length": len(text),
        }


async def benchmark_text_length(
    length: int,
    samples: int,
    warmup: int,
    tts_service: ChatterboxTTSService,
) -> dict[str, Any]:
    """Benchmark TTS for a specific text length.

    Args:
        length: Text length in characters.
        samples: Number of samples.
        warmup: Number of warmup samples.
        tts_service: TTS service instance.

    Returns:
        Benchmark results.
    """
    text = generate_test_text(length)
    logger.info("benchmarking_text_length", length=length, text=text[:50])

    # Warmup
    if warmup > 0:
        logger.info("warming_up", samples=warmup)
        for _ in range(warmup):
            await benchmark_single(text[:min(length, 50)], tts_service)

    # Benchmark
    results = []
    for i in range(samples):
        result = await benchmark_single(text, tts_service)
        results.append(result)

        if i % 10 == 0:
            logger.info(
                "benchmark_progress",
                length=length,
                sample=i + 1,
                total=samples,
            )

    # Calculate statistics
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    if not successful:
        return {
            "length": length,
            "samples": samples,
            "successful": 0,
            "failed": len(failed),
            "error_rate": 1.0,
        }

    latencies = [r["latency"] for r in successful]
    rtfs = [r["rtf"] for r in successful]

    return {
        "length": length,
        "samples": samples,
        "successful": len(successful),
        "failed": len(failed),
        "error_rate": len(failed) / samples,
        "latency": {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "p95": sorted(latencies)[int(len(latencies) * 0.95)],
            "p99": sorted(latencies)[int(len(latencies) * 0.99)],
        },
        "rtf": {
            "mean": statistics.mean(rtfs),
            "median": statistics.median(rtfs),
            "min": min(rtfs),
            "max": max(rtfs),
        },
    }


async def run_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run the benchmark.

    Args:
        args: Parsed command line arguments.

    Returns:
        Benchmark results.
    """
    # Initialize TTS service
    if args.provider == "mock":
        from hermes.services.tts import MockTTSService

        tts_service = MockTTSService(duration_seconds=0.1)
    else:
        tts_service = ChatterboxTTSService()

    logger.info(
        "starting_benchmark",
        provider=args.provider,
        samples=args.samples,
        text_lengths=args.text_lengths,
    )

    results = []
    for length in args.text_lengths:
        result = await benchmark_text_length(
            length=length,
            samples=args.samples,
            warmup=args.warmup,
            tts_service=tts_service,
        )
        results.append(result)

        # Print intermediate results
        if result["successful"] > 0:
            latency = result["latency"]
            print(f"\nResults for {length} characters:")
            print(f"  Success rate: {result['successful']}/{result['samples']} "
                  f"({(1 - result['error_rate']) * 100:.1f}%)")
            print(f"  Latency: {latency['mean']:.3f}s "
                  f"(min: {latency['min']:.3f}s, max: {latency['max']:.3f}s)")
            print(f"  P95: {latency['p95']:.3f}s, P99: {latency['p99']:.3f}s")
            print(f"  RTF: {result['rtf']['mean']:.3f}x")

    return results


async def main() -> int:
    """Main entry point.

    Returns:
        Exit code.
    """
    args = parse_args()

    try:
        results = await run_benchmark(args)

        # Print summary
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        for result in results:
            if result.get("successful", 0) > 0:
                latency = result["latency"]
                print(f"\nText length: {result['length']} characters")
                print(f"  Mean latency: {latency['mean']:.3f}s")
                print(f"  P95 latency:  {latency['p95']:.3f}s")
                print(f"  Mean RTF:     {result['rtf']['mean']:.3f}x")

        # Save results if requested
        if args.output:
            import json

            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")

        return 0

    except Exception as e:
        logger.exception("benchmark_failed", error=str(e))
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
