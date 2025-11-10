"""
Example Python client for testing WebSocket live transcription.
Reads an audio file and streams it to the WebSocket endpoint.
"""
import asyncio
import websockets
import numpy as np
import soundfile as sf
import json
import sys


async def stream_audio_file(audio_path: str, ws_url: str = "ws://localhost:8000/ws/transcribe"):
    """
    Stream audio file to WebSocket endpoint.

    Args:
        audio_path: Path to audio file (WAV, MP3, etc.)
        ws_url: WebSocket URL
    """
    print(f"Loading audio file: {audio_path}")

    # Load audio file
    try:
        audio_data, sample_rate = sf.read(audio_path)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        print(f"Resampling from {sample_rate}Hz to 16000Hz...")
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    # Ensure float32
    audio_data = audio_data.astype(np.float32)

    print(f"Audio duration: {len(audio_data) / sample_rate:.2f} seconds")
    print(f"Connecting to {ws_url}...")

    try:
        async with websockets.connect(ws_url) as websocket:
            print("Connected! Streaming audio...")

            # Send audio in chunks (simulate real-time streaming)
            chunk_size = int(sample_rate * 0.5)  # 500ms chunks
            total_chunks = len(audio_data) // chunk_size

            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]

                # Send chunk
                await websocket.send(chunk.tobytes())

                # Listen for responses (non-blocking)
                try:
                    while True:
                        response = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                        result = json.loads(response)

                        if 'error' in result:
                            print(f"\n❌ Error: {result['error']}")
                        else:
                            segment = result['segment']
                            print(f"\n[{segment['speaker']}] {segment['start']:.2f}s - {segment['end']:.2f}s")
                            print(f"  {result['original_language'].upper()}: {segment['text']}")
                            print(f"  {result['target_language'].upper()}: {segment['translation']}")

                except asyncio.TimeoutError:
                    pass  # No response yet

                # Progress indicator
                chunk_num = i // chunk_size + 1
                print(f"\rStreaming: {chunk_num}/{total_chunks} chunks", end='')

                # Small delay to simulate real-time
                await asyncio.sleep(0.1)

            # Send empty buffer to signal end
            print("\n\nSending end signal...")
            await websocket.send(b'')

            # Wait for final responses
            print("Waiting for final transcriptions...")
            try:
                while True:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    result = json.loads(response)

                    if 'error' in result:
                        print(f"\n❌ Error: {result['error']}")
                    else:
                        segment = result['segment']
                        print(f"\n[{segment['speaker']}] {segment['start']:.2f}s - {segment['end']:.2f}s")
                        print(f"  {result['original_language'].upper()}: {segment['text']}")
                        print(f"  {result['target_language'].upper()}: {segment['translation']}")

            except asyncio.TimeoutError:
                print("\nDone!")

    except websockets.exceptions.WebSocketException as e:
        print(f"WebSocket error: {e}")
    except Exception as e:
        print(f"Error: {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python example_client.py <audio_file>")
        print("\nExample:")
        print("  python example_client.py test_audio.wav")
        sys.exit(1)

    audio_file = sys.argv[1]
    ws_url = sys.argv[2] if len(sys.argv) > 2 else "ws://localhost:8000/ws/transcribe"

    # Run async function
    asyncio.run(stream_audio_file(audio_file, ws_url))


if __name__ == "__main__":
    main()
