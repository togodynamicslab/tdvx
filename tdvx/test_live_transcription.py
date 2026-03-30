"""
Test script for live transcription WebSocket endpoint.
Sends audio file to the WebSocket and saves the transcription results.

Usage:
    python test_live_transcription.py <audio_file_path>

Example:
    python test_live_transcription.py C:/Users/Matheus/Downloads/audio3.mp3
"""

import asyncio
import websockets
import json
import sys
import librosa
import numpy as np
from pathlib import Path
from datetime import datetime


async def test_live_transcription(audio_file_path: str):
    """
    Test live transcription by sending audio file through WebSocket.

    Args:
        audio_file_path: Path to audio file (WAV, MP3, etc.)
    """
    # Load audio file
    print(f"\n[1/4] Loading audio file: {audio_file_path}")
    try:
        # Load and resample to 16kHz mono (required format)
        audio_data, sr = librosa.load(audio_file_path, sr=16000, mono=True)
        print(f"✓ Loaded {len(audio_data)/sr:.2f}s of audio at {sr}Hz")
    except Exception as e:
        print(f"✗ Error loading audio: {e}")
        return

    # Connect to WebSocket
    ws_url = "ws://localhost:8000/ws/transcribe"
    print(f"\n[2/4] Connecting to WebSocket: {ws_url}")

    results = []

    try:
        async with websockets.connect(ws_url) as websocket:
            print("✓ Connected to WebSocket")

            # Send audio in chunks (simulate real-time streaming)
            print(f"\n[3/4] Sending audio in chunks...")

            # Chunk size: 4096 samples (same as browser)
            chunk_size = 4096
            total_chunks = len(audio_data) // chunk_size

            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i+chunk_size]

                # Convert to float32 (same format as browser sends)
                float32_chunk = chunk.astype(np.float32)

                # Send to WebSocket
                await websocket.send(float32_chunk.tobytes())

                # Progress indicator
                chunk_num = i // chunk_size + 1
                if chunk_num % 10 == 0:
                    print(f"  Sent {chunk_num}/{total_chunks} chunks ({i/sr:.1f}s / {len(audio_data)/sr:.1f}s)")

                # Small delay to simulate real-time (optional)
                await asyncio.sleep(0.01)

                # Check for responses (non-blocking)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.001)
                    data = json.parse(response)
                    results.append(data)

                    # Print received transcription
                    if 'segment' in data:
                        seg = data['segment']
                        print(f"\n  📝 [{seg.get('speaker', 'SPEAKER_00')}] {seg.get('text', '')}")
                        if seg.get('translation'):
                            print(f"     → {seg['translation']}")
                except asyncio.TimeoutError:
                    # No response yet, continue sending
                    pass
                except json.JSONDecodeError as e:
                    print(f"  ⚠ JSON decode error: {e}")

            print(f"\n✓ Sent all {total_chunks} chunks")

            # Send end signal
            print("\n[4/4] Sending end signal and waiting for final results...")
            await websocket.send("end")

            # Collect remaining responses (with timeout)
            try:
                while True:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(response)
                    results.append(data)

                    # Print received transcription
                    if 'segment' in data:
                        seg = data['segment']
                        print(f"\n  📝 [{seg.get('speaker', 'SPEAKER_00')}] {seg.get('text', '')}")
                        if seg.get('translation'):
                            print(f"     → {seg['translation']}")
                    elif 'error' in data:
                        print(f"\n  ✗ Error: {data['error']}")

            except asyncio.TimeoutError:
                print("\n✓ No more responses (timeout)")
            except websockets.exceptions.ConnectionClosed:
                print("\n✓ WebSocket connection closed")

    except Exception as e:
        print(f"\n✗ WebSocket error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Save results to JSON file
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = Path(audio_file_path).stem
        output_file = f"test_results/live_{audio_filename}_{timestamp}.json"

        Path("test_results").mkdir(exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'audio_file': audio_file_path,
                'duration_seconds': len(audio_data) / sr,
                'chunks_sent': total_chunks,
                'results_received': len(results),
                'test_timestamp': timestamp,
                'results': results
            }, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Saved {len(results)} results to: {output_file}")

        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Audio duration: {len(audio_data)/sr:.2f}s")
        print(f"Chunks sent: {total_chunks}")
        print(f"Results received: {len(results)}")

        # Extract segments
        segments = [r['segment'] for r in results if 'segment' in r]
        if segments:
            print(f"Transcription segments: {len(segments)}")
            print("\nFull Transcription:")
            print("-"*60)
            for seg in segments:
                speaker = seg.get('speaker', 'SPEAKER_00')
                text = seg.get('text', '')
                translation = seg.get('translation', '')
                print(f"[{speaker}] {text}")
                if translation and translation != text:
                    print(f"           → {translation}")

        print("="*60)
    else:
        print("\n⚠ No results received")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_live_transcription.py <audio_file_path>")
        print("\nExample:")
        print("  python test_live_transcription.py C:/Users/Matheus/Downloads/audio3.mp3")
        sys.exit(1)

    audio_file = sys.argv[1]

    if not Path(audio_file).exists():
        print(f"Error: File not found: {audio_file}")
        sys.exit(1)

    print("="*60)
    print("LIVE TRANSCRIPTION TEST")
    print("="*60)

    asyncio.run(test_live_transcription(audio_file))
