"""
Benchmark script to compare TDv1 and TDv1-Fast pipelines.

Usage:
    python benchmark.py <audio_file_path>

Example:
    python benchmark.py test_audio.wav
"""

import sys
import time
import argparse
from pathlib import Path
import json
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.processor import processor
from app.services.whisper_service import get_or_create_whisper_service
from app.models.model_config import ModelType
import librosa
import numpy as np


def load_audio(file_path: str, sample_rate: int = 16000) -> tuple[np.ndarray, float]:
    """Load audio file and return audio data and duration."""
    print(f"Loading audio from: {file_path}")
    audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    duration = len(audio) / sample_rate
    print(f"Audio loaded: {duration:.2f}s, sample rate: {sr}Hz")
    return audio, duration


def benchmark_model(audio_data: np.ndarray, sample_rate: int, model_type: str, audio_duration: float) -> dict:
    """Benchmark a single model pipeline."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_type.upper()}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        result = processor.process_audio(
            audio_data=audio_data,
            sample_rate=sample_rate,
            model_type=model_type
        )

        processing_time = time.time() - start_time
        rtf = processing_time / audio_duration if audio_duration > 0 else 0
        speed_multiplier = 1 / rtf if rtf > 0 else 0

        # Extract metrics
        num_segments = len(result.segments)
        total_words = sum(len(seg.text.split()) for seg in result.segments)
        speakers = set(seg.speaker for seg in result.segments)

        metrics = {
            "model": model_type,
            "audio_duration": audio_duration,
            "processing_time": processing_time,
            "rtf": rtf,
            "speed_multiplier": speed_multiplier,
            "num_segments": num_segments,
            "num_speakers": len(speakers),
            "total_words": total_words,
            "original_language": result.original_language,
            "target_language": result.target_language,
            "success": True,
            "segments": [
                {
                    "speaker": seg.speaker,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "translation": seg.translation
                }
                for seg in result.segments
            ]
        }

        print(f"\n[OK] Processing completed successfully!")
        print(f"  Audio duration: {audio_duration:.2f}s")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Real-Time Factor (RTF): {rtf:.3f}x")
        print(f"  Speed: {speed_multiplier:.2f}x faster than real-time" if rtf < 1 else f"  Speed: {rtf:.2f}x slower than real-time")
        print(f"  Segments: {num_segments}")
        print(f"  Speakers: {len(speakers)}")
        print(f"  Total words: {total_words}")

        return metrics

    except Exception as e:
        processing_time = time.time() - start_time
        print(f"\n[FAIL] Processing failed after {processing_time:.2f}s")
        print(f"  Error: {str(e)}")

        return {
            "model": model_type,
            "audio_duration": audio_duration,
            "processing_time": processing_time,
            "success": False,
            "error": str(e)
        }


def compare_results(tdv1_result: dict, tdv1_fast_result: dict):
    """Print comparison between the two models."""
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}\n")

    # Performance comparison
    print("Performance:")
    print(f"  TDv1:      {tdv1_result['processing_time']:.2f}s (RTF: {tdv1_result['rtf']:.3f}x)")
    print(f"  TDv1-Fast: {tdv1_fast_result['processing_time']:.2f}s (RTF: {tdv1_fast_result['rtf']:.3f}x)")

    if tdv1_result['success'] and tdv1_fast_result['success']:
        speedup = tdv1_result['processing_time'] / tdv1_fast_result['processing_time']
        print(f"\n  Speedup: TDv1-Fast is {speedup:.2f}x faster than TDv1")

    # Accuracy comparison
    if tdv1_result.get('success') and tdv1_fast_result.get('success'):
        print(f"\nOutput:")
        print(f"  TDv1:      {tdv1_result['num_segments']} segments, {tdv1_result['total_words']} words")
        print(f"  TDv1-Fast: {tdv1_fast_result['num_segments']} segments, {tdv1_fast_result['total_words']} words")

        # Speaker comparison
        print(f"\nSpeakers:")
        print(f"  TDv1:      {tdv1_result['num_speakers']}")
        print(f"  TDv1-Fast: {tdv1_fast_result['num_speakers']}")

    # Recommendation
    print(f"\n{'='*60}")
    print("RECOMMENDATION:")
    print(f"{'='*60}")
    if tdv1_result.get('success') and tdv1_fast_result.get('success'):
        if tdv1_fast_result['rtf'] < 1.0:
            print("[OK] TDv1-Fast achieves real-time processing (RTF < 1.0)")
            print("  Recommended for: Live transcription")
        else:
            print("[WARN] TDv1-Fast does not achieve real-time (RTF > 1.0)")
            print("  Consider: Hardware upgrade or shorter audio chunks")

        print(f"\n[OK] TDv1 provides {tdv1_result['rtf']:.1f}x processing")
        print("  Recommended for: High-quality file transcription")


def save_results(tdv1_result: dict, tdv1_fast_result: dict, output_file: str):
    """Save benchmark results to JSON file."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "tdv1": tdv1_result,
        "tdv1_fast": tdv1_fast_result
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TDv1 and TDv1-Fast transcription pipelines"
    )
    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to audio file to benchmark"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="benchmark_results.json",
        help="Output file for results (default: benchmark_results.json)"
    )
    parser.add_argument(
        "--tdv1-only",
        action="store_true",
        help="Only benchmark TDv1"
    )
    parser.add_argument(
        "--tdv1-fast-only",
        action="store_true",
        help="Only benchmark TDv1-Fast"
    )

    args = parser.parse_args()

    # Validate audio file exists
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {args.audio_file}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("TRANSCRIPTION PIPELINE BENCHMARK")
    print(f"{'='*60}")
    print(f"Audio file: {audio_path.name}")
    print(f"Full path: {audio_path.absolute()}")

    # Load audio
    try:
        audio_data, audio_duration = load_audio(str(audio_path))
    except Exception as e:
        print(f"\nError loading audio: {e}")
        sys.exit(1)

    # Pre-load models
    print(f"\nPre-loading models...")
    if not args.tdv1_fast_only:
        tdv1_svc = get_or_create_whisper_service(ModelType.TDV1)
        tdv1_svc.load_model()
        print(f"[OK] TDv1 model loaded")

    if not args.tdv1_only:
        tdv1_fast_svc = get_or_create_whisper_service(ModelType.TDV1_FAST)
        tdv1_fast_svc.load_model()
        print(f"[OK] TDv1-Fast model loaded")

    # Benchmark models
    results = {}

    if not args.tdv1_fast_only:
        results['tdv1'] = benchmark_model(audio_data, 16000, ModelType.TDV1, audio_duration)

    if not args.tdv1_only:
        results['tdv1_fast'] = benchmark_model(audio_data, 16000, ModelType.TDV1_FAST, audio_duration)

    # Compare results
    if 'tdv1' in results and 'tdv1_fast' in results:
        compare_results(results['tdv1'], results['tdv1_fast'])

    # Save results
    save_results(
        results.get('tdv1', {}),
        results.get('tdv1_fast', {}),
        args.output
    )

    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
