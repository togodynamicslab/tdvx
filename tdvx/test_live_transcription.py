"""
test_live_transcription.py — Testa o WebSocket /ws/transcribe do TDvX v3.

Uso:
    python test_live_transcription.py <audio_file>
    python test_live_transcription.py audio.wav --url ws://localhost:8000/ws/transcribe
"""
import asyncio
import json
import sys
import argparse
import librosa
import numpy as np
import websockets
from datetime import datetime
from pathlib import Path


async def test_live_transcription(audio_file_path: str, ws_url: str):
    print(f"\n[1/4] Carregando audio: {audio_file_path}")
    try:
        audio_data, sr = librosa.load(audio_file_path, sr=16000, mono=True)
        print(f"  OK  {len(audio_data)/sr:.2f}s @ {sr}Hz")
    except Exception as e:
        print(f"  ERRO ao carregar audio: {e}")
        return

    print(f"\n[2/4] Conectando: {ws_url}")
    results = []

    try:
        async with websockets.connect(ws_url) as websocket:
            print("  OK  Conectado")

            chunk_size = 4096
            total_chunks = max(1, len(audio_data) // chunk_size)
            print(f"\n[3/4] Enviando {total_chunks} chunks de {chunk_size} samples (256ms cada)...")

            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i+chunk_size].astype(np.float32)
                await websocket.send(chunk.tobytes())

                chunk_num = i // chunk_size + 1
                if chunk_num % 10 == 0:
                    print(f"  {chunk_num}/{total_chunks} chunks ({i/sr:.1f}s/{len(audio_data)/sr:.1f}s)")

                await asyncio.sleep(0.01)

                # Recebe respostas parciais (non-blocking)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.001)
                    data = json.loads(response)
                    results.append(data)
                    if 'segment' in data:
                        seg = data['segment']
                        print(f"\n  [{seg.get('speaker')}] {seg.get('text')}")
                except asyncio.TimeoutError:
                    pass

            print(f"\n  OK  Todos os {total_chunks} chunks enviados")

            # Sinaliza fim de stream com bytes vazios
            print(f"\n[4/4] Sinalizando fim de stream...")
            await websocket.send(b'')

            # Coleta respostas finais
            try:
                while True:
                    response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    data = json.loads(response)
                    results.append(data)
                    if 'segment' in data:
                        seg = data['segment']
                        print(f"  [{seg.get('speaker')}] {seg.get('start', 0):.1f}s  {seg.get('text')}")
                    elif 'error' in data:
                        print(f"  ERRO: {data['error']}")
            except asyncio.TimeoutError:
                print("  OK  Sem mais respostas")
            except websockets.exceptions.ConnectionClosed:
                print("  OK  Conexao fechada pelo servidor")

    except Exception as e:
        print(f"\n  ERRO WebSocket: {e}")
        import traceback; traceback.print_exc()
        return

    # Resumo
    segments = [r['segment'] for r in results if 'segment' in r]
    print(f"\n{'='*60}")
    print(f"  Audio:      {len(audio_data)/sr:.2f}s")
    print(f"  Segmentos:  {len(segments)}")
    print(f"  Respostas:  {len(results)}")

    if segments:
        print(f"\n  Transcricao completa:")
        print(f"  {'-'*50}")
        for seg in segments:
            print(f"  [{seg.get('speaker')}] {seg.get('start', 0):.1f}s-{seg.get('end', 0):.1f}s  {seg.get('text')}")
    print(f"{'='*60}")

    # Salva resultado
    if results:
        out_dir = Path("test_results")
        out_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = out_dir / f"live_{Path(audio_file_path).stem}_{ts}.json"
        out_file.write_text(
            json.dumps({"audio": audio_file_path, "segments": segments, "raw": results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n  Salvo: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testa WebSocket /ws/transcribe do TDvX")
    parser.add_argument("audio", help="Arquivo de audio (WAV, MP3, etc.)")
    parser.add_argument("--url", default="ws://localhost:8000/ws/transcribe", help="URL do WebSocket")
    args = parser.parse_args()

    if not Path(args.audio).exists():
        print(f"Arquivo nao encontrado: {args.audio}")
        sys.exit(1)

    print("="*60)
    print("LIVE TRANSCRIPTION TEST — TDvX v3")
    print("="*60)
    asyncio.run(test_live_transcription(args.audio, args.url))
