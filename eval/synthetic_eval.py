#!/usr/bin/env python3
"""
synthetic_eval.py — Avaliação do modelo com áudios sintéticos e ruídos
=======================================================================

Pipeline:
  1. Gera áudios limpos a partir de frases pt-BR via edge-tts (TTS neural)
  2. Aplica variantes de ruído (white noise, reverb, babble, low-bitrate…)
  3. Transcreve cada variante com o modelo Whisper (fine-tuned ou base)
  4. Compara transcrição vs. texto original (WER)
  5. Gera relatório JSON + CSV + tabela no terminal
  6. Registra métricas e artefatos no MLflow (opcional)

Uso:
    # Usar modelo base whisper-medium
    python eval/synthetic_eval.py

    # Usar modelo fine-tunado (caminho local)
    python eval/synthetic_eval.py --model-path ./models/tdvx-v4-cv-pt-ct2

    # Com MLflow local
    python eval/synthetic_eval.py --mlflow-uri ./mlruns

    # Com MLflow remoto
    python eval/synthetic_eval.py --mlflow-uri http://localhost:5000 --mlflow-experiment tdvx-eval

    # Salvar áudios gerados para inspeção manual
    python eval/synthetic_eval.py --save-audio
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ─────────────────────────────────────────────────────────────────────────────
# Imports opcionais com mensagens amigáveis
# ─────────────────────────────────────────────────────────────────────────────

try:
    import soundfile as sf
except ImportError:
    sys.exit("pip install soundfile")

try:
    import librosa
except ImportError:
    sys.exit("pip install librosa")

try:
    import edge_tts
except ImportError:
    sys.exit("pip install edge-tts")

try:
    from faster_whisper import WhisperModel
except ImportError:
    sys.exit("pip install faster-whisper")

try:
    from jiwer import wer as compute_wer
except ImportError:
    sys.exit("pip install jiwer")

_MLFLOW_AVAILABLE = False
try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Frases padrão pt-BR (usadas quando --texts-file não é passado)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TEXTS = [
    "Bom dia, como você está hoje?",
    "O projeto de transcrição está funcionando muito bem.",
    "Precisamos agendar uma reunião para amanhã de manhã.",
    "O relatório financeiro do trimestre foi concluído.",
    "Por favor, envie o documento por e-mail assim que possível.",
    "A inteligência artificial está transformando o mercado de trabalho.",
    "Eu gostaria de falar com o responsável pelo setor técnico.",
    "O sistema de reconhecimento de voz precisa ser calibrado.",
    "As transações foram processadas com sucesso pelo banco.",
    "Qual é o prazo de entrega para esse pedido urgente?",
    "O médico recomendou repouso e hidratação abundante.",
    "Vamos revisar o contrato antes de assinar qualquer documento.",
    "A conferência internacional ocorrerá no mês de setembro.",
    "Poderia repetir mais devagar, por favor?",
    "Os dados foram atualizados no sistema automaticamente.",
]

# ─────────────────────────────────────────────────────────────────────────────
# Tipos de ruído
# ─────────────────────────────────────────────────────────────────────────────

NOISE_TYPES = [
    "clean",
    "white_noise_high",    # SNR ~20 dB — ruído baixo
    "white_noise_mid",     # SNR ~10 dB — ruído moderado
    "white_noise_low",     # SNR ~5 dB  — ruído alto
    "pink_noise",          # ruído rosa (simula fundo ambiente)
    "reverb_small",        # reverb de sala pequena
    "reverb_large",        # reverb de sala grande / eco
    "low_bitrate",         # compressão de áudio (telefone)
    "speed_slow",          # fala 20% mais lenta
    "speed_fast",          # fala 20% mais rápida
    "pitch_down",          # pitch -3 semitons
    "pitch_up",            # pitch +3 semitons
]

SAMPLE_RATE = 16_000   # Whisper espera 16kHz


# ─────────────────────────────────────────────────────────────────────────────
# Estrutura de resultado
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    text_id: int
    original_text: str
    noise_type: str
    transcription: str
    wer: float
    inference_time_s: float
    language_detected: str
    avg_log_prob: float
    no_speech_prob: float
    label: str = field(init=False)

    def __post_init__(self):
        self.label = self._classify()

    def _classify(self) -> str:
        if self.no_speech_prob > 0.6:
            return "SILENCIO_DETECTADO"
        if self.wer == 0.0:
            return "PERFEITO"
        if self.wer <= 0.1:
            return "EXCELENTE"
        if self.wer <= 0.25:
            return "BOM"
        if self.wer <= 0.5:
            return "RUIM"
        return "INCOMPREENSIVEL"


# ─────────────────────────────────────────────────────────────────────────────
# Geração de áudio via edge-tts (TTS neural pt-BR)
# ─────────────────────────────────────────────────────────────────────────────

async def _tts_to_bytes(text: str, voice: str = "pt-BR-FranciscaNeural") -> bytes:
    communicate = edge_tts.Communicate(text, voice)
    audio_bytes = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes += chunk["data"]
    return audio_bytes


def generate_clean_audio(text: str, voice: str = "pt-BR-FranciscaNeural") -> np.ndarray:
    mp3_bytes = asyncio.run(_tts_to_bytes(text, voice))

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(mp3_bytes)
        tmp_path = f.name

    try:
        audio, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
    finally:
        os.unlink(tmp_path)

    return audio.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Aplicadores de ruído
# ─────────────────────────────────────────────────────────────────────────────

def _add_noise(audio: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    signal_power = np.mean(audio ** 2) + 1e-9
    noise_power  = np.mean(noise ** 2) + 1e-9
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    scaled_noise = noise * np.sqrt(target_noise_power / noise_power)

    if len(scaled_noise) < len(audio):
        reps = int(np.ceil(len(audio) / len(scaled_noise)))
        scaled_noise = np.tile(scaled_noise, reps)
    scaled_noise = scaled_noise[: len(audio)]

    return np.clip(audio + scaled_noise, -1.0, 1.0)


def apply_noise(audio: np.ndarray, noise_type: str) -> np.ndarray:
    n = len(audio)

    if noise_type == "clean":
        return audio

    elif noise_type == "white_noise_high":
        return _add_noise(audio, np.random.randn(n).astype(np.float32), snr_db=20)

    elif noise_type == "white_noise_mid":
        return _add_noise(audio, np.random.randn(n).astype(np.float32), snr_db=10)

    elif noise_type == "white_noise_low":
        return _add_noise(audio, np.random.randn(n).astype(np.float32), snr_db=5)

    elif noise_type == "pink_noise":
        white = np.random.randn(n).astype(np.float32)
        pink  = np.cumsum(white)
        pink  = pink / (np.max(np.abs(pink)) + 1e-9)
        return _add_noise(audio, pink, snr_db=12)

    elif noise_type == "reverb_small":
        ir_len = int(0.2 * SAMPLE_RATE)
        ir = (np.exp(-np.linspace(0, 6, ir_len)) * np.random.randn(ir_len)).astype(np.float32)
        ir /= np.max(np.abs(ir)) + 1e-9
        reverbed = np.convolve(audio.astype(np.float32), ir, mode="full")[:n].astype(np.float32)
        return np.clip(reverbed / (np.max(np.abs(reverbed)) + 1e-9), -1.0, 1.0).astype(np.float32)

    elif noise_type == "reverb_large":
        ir_len = int(1.5 * SAMPLE_RATE)
        ir = (np.exp(-np.linspace(0, 3, ir_len)) * np.random.randn(ir_len)).astype(np.float32)
        ir /= np.max(np.abs(ir)) + 1e-9
        reverbed = np.convolve(audio.astype(np.float32), ir, mode="full")[:n].astype(np.float32)
        return np.clip(reverbed / (np.max(np.abs(reverbed)) + 1e-9), -1.0, 1.0).astype(np.float32)

    elif noise_type == "low_bitrate":
        low_sr = 8000
        degraded = librosa.resample(audio, orig_sr=SAMPLE_RATE, target_sr=low_sr)
        restored = librosa.resample(degraded, orig_sr=low_sr, target_sr=SAMPLE_RATE)
        restored = restored[:n] if len(restored) >= n else np.pad(restored, (0, n - len(restored)))
        return restored.astype(np.float32)

    elif noise_type == "speed_slow":
        stretched = librosa.effects.time_stretch(audio, rate=0.8)
        stretched = stretched[:n] if len(stretched) >= n else np.pad(stretched, (0, n - len(stretched)))
        return stretched.astype(np.float32)

    elif noise_type == "speed_fast":
        stretched = librosa.effects.time_stretch(audio, rate=1.2)
        stretched = stretched[:n] if len(stretched) >= n else np.pad(stretched, (0, n - len(stretched)))
        return stretched.astype(np.float32)

    elif noise_type == "pitch_down":
        return librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=-3).astype(np.float32)

    elif noise_type == "pitch_up":
        return librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=3).astype(np.float32)

    return audio


# ─────────────────────────────────────────────────────────────────────────────
# Transcrição com faster-whisper
# ─────────────────────────────────────────────────────────────────────────────

def transcribe(model: "WhisperModel", audio: np.ndarray) -> Tuple[str, str, float, float]:
    """Retorna (texto, idioma, avg_log_prob, no_speech_prob)."""
    segments, info = model.transcribe(
        audio,
        language="pt",
        beam_size=5,
        vad_filter=True,
    )
    texts = []
    avg_log_probs = []
    no_speech_probs = []

    for seg in segments:
        texts.append(seg.text.strip())
        avg_log_probs.append(seg.avg_logprob)
        no_speech_probs.append(seg.no_speech_prob)

    avg_log_prob = float(np.mean(avg_log_probs)) if avg_log_probs else 0.0
    no_speech_prob = float(np.mean(no_speech_probs)) if no_speech_probs else 0.0

    return " ".join(texts).strip(), info.language, avg_log_prob, no_speech_prob


def _safe_wer(reference: str, hypothesis: str) -> float:
    if not reference.strip():
        return 1.0
    if not hypothesis.strip():
        return 1.0
    try:
        return float(compute_wer(reference.lower(), hypothesis.lower()))
    except Exception:
        return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Relatório
# ─────────────────────────────────────────────────────────────────────────────

LABEL_COLOR = {
    "PERFEITO":           "\033[92m",
    "EXCELENTE":          "\033[96m",
    "BOM":                "\033[93m",
    "RUIM":               "\033[91m",
    "INCOMPREENSIVEL":    "\033[35m",
    "SILENCIO_DETECTADO": "\033[90m",
}
RESET = "\033[0m"


def _color(label: str, text: str) -> str:
    return f"{LABEL_COLOR.get(label, '')}{text}{RESET}"


def print_table(results: List[EvalResult]):
    header = f"{'#':>3} {'Ruído':<22} {'WER':>6} {'Label':<22} {'Transcrição'}"
    print("\n" + "=" * 110)
    print(header)
    print("-" * 110)

    for r in results:
        wer_str = f"{r.wer:.1%}"
        label_col = _color(r.label, f"{r.label:<22}")
        transcription_col = r.transcription[:60] + ("…" if len(r.transcription) > 60 else "")
        print(f"{r.text_id:>3} {r.noise_type:<22} {wer_str:>6} {label_col} {transcription_col}")

    print("=" * 110)


def save_reports(results: List[EvalResult], output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = output_dir / f"eval_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

    csv_path = output_dir / f"eval_{ts}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in results])

    summary: Dict = {}
    for r in results:
        if r.noise_type not in summary:
            summary[r.noise_type] = {"wer_total": 0.0, "count": 0, "labels": {}}
        summary[r.noise_type]["wer_total"] += r.wer
        summary[r.noise_type]["count"] += 1
        summary[r.noise_type]["labels"][r.label] = summary[r.noise_type]["labels"].get(r.label, 0) + 1

    print("\nSumário por tipo de ruído:")
    print(f"{'Ruído':<25} {'WER médio':>10} {'Distribuição de labels'}")
    print("-" * 80)
    for noise, data in sorted(summary.items()):
        avg = data["wer_total"] / data["count"]
        labels_str = " | ".join(f"{k}:{v}" for k, v in data["labels"].items())
        print(f"{noise:<25} {avg:>9.1%}  {labels_str}")

    log.info("Relatório salvo em: %s e %s", json_path, csv_path)
    return json_path, csv_path


# ─────────────────────────────────────────────────────────────────────────────
# MLflow tracking
# ─────────────────────────────────────────────────────────────────────────────

class MlflowTracker:
    """Wrapper opcional de MLflow — silencioso se MLflow não estiver instalado."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._enabled = False
        self._run = None

        if not _MLFLOW_AVAILABLE:
            log.info("MLflow não instalado — tracking desabilitado. (pip install mlflow)")
            return

        uri = args.mlflow_uri or os.environ.get("MLFLOW_TRACKING_URI", "")
        if not uri:
            log.info("MLflow: --mlflow-uri não informado — tracking desabilitado.")
            return

        experiment = args.mlflow_experiment or os.environ.get(
            "MLFLOW_EVAL_EXPERIMENT", "tdvx-synthetic-eval"
        )
        run_name = args.run_name or f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        try:
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment)
            self._run = mlflow.start_run(run_name=run_name)
            self._enabled = True
            log.info("MLflow: run '%s' (id=%s) | URI: %s", run_name, self._run.info.run_id, uri)
        except Exception as exc:
            # Tenta fallback local antes de desistir
            local_uri = "./mlruns"
            log.warning(
                "MLflow: falha ao conectar em '%s' (%s). Tentando fallback local em %s…",
                uri, exc, local_uri,
            )
            try:
                mlflow.set_tracking_uri(local_uri)
                mlflow.set_experiment(experiment)
                self._run = mlflow.start_run(run_name=run_name)
                self._enabled = True
                log.info("MLflow (local): run '%s' em %s", run_name, local_uri)
            except Exception as exc2:
                log.warning("MLflow: fallback local também falhou (%s) — tracking desabilitado.", exc2)

    def __enter__(self) -> "MlflowTracker":
        return self

    def __exit__(self, *_) -> None:
        if self._enabled and self._run:
            mlflow.end_run()

    # ── Params ────────────────────────────────────────────────────────────────

    def log_params(self, args: argparse.Namespace, num_texts: int) -> None:
        if not self._enabled:
            return
        mlflow.log_params({
            "model_path":   args.model_path,
            "device":       args.device,
            "tts_voice":    args.tts_voice,
            "num_texts":    num_texts,
            "noise_types":  ",".join(args.noise_types),
            "num_noise_types": len(args.noise_types),
        })

    # ── Métricas finais ────────────────────────────────────────────────────────

    def log_metrics(self, results: List[EvalResult]) -> None:
        if not self._enabled:
            return

        # WER e tempo médio por tipo de ruído
        by_noise: Dict[str, Dict] = {}
        for r in results:
            entry = by_noise.setdefault(r.noise_type, {"wers": [], "times": []})
            entry["wers"].append(r.wer)
            entry["times"].append(r.inference_time_s)

        metrics: Dict[str, float] = {}
        for noise, data in by_noise.items():
            metrics[f"wer_{noise}"]            = round(float(np.mean(data["wers"])), 4)
            metrics[f"inference_time_{noise}"] = round(float(np.mean(data["times"])), 4)

        # Métricas globais
        all_wers  = [r.wer for r in results]
        all_times = [r.inference_time_s for r in results]
        metrics["wer_overall"]            = round(float(np.mean(all_wers)), 4)
        metrics["wer_clean"]              = metrics.get("wer_clean", float("nan"))
        metrics["inference_time_mean_s"]  = round(float(np.mean(all_times)), 4)

        # Distribuição de labels (%)
        label_counts: Dict[str, int] = {}
        for r in results:
            label_counts[r.label] = label_counts.get(r.label, 0) + 1
        total = len(results)
        for label, count in label_counts.items():
            metrics[f"pct_{label.lower()}"] = round(count / total * 100, 2)

        mlflow.log_metrics(metrics)
        log.info("MLflow: %d métricas registradas.", len(metrics))

    # ── Artefatos ─────────────────────────────────────────────────────────────

    def log_artifacts(self, json_path: Path, csv_path: Path) -> None:
        if not self._enabled:
            return
        mlflow.log_artifact(str(json_path), artifact_path="eval_reports")
        mlflow.log_artifact(str(csv_path),  artifact_path="eval_reports")
        log.info("MLflow: artefatos registrados (%s, %s).", json_path.name, csv_path.name)

    # ── Tags ──────────────────────────────────────────────────────────────────

    def set_tags(self, model_path: str, device: str) -> None:
        if not self._enabled:
            return
        mlflow.set_tags({
            "model": model_path,
            "device": device,
            "task": "synthetic-eval-pt-br",
        })


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Avaliação sintética do modelo TDvX")
    p.add_argument("--model-path", default="Systran/faster-whisper-medium",
                   help="Caminho do modelo (local ou HuggingFace CTranslate2)")
    p.add_argument("--texts-file", type=Path,
                   help="Arquivo .txt com frases (uma por linha)")
    p.add_argument("--noise-types", nargs="+", default=NOISE_TYPES,
                   help="Tipos de ruído a testar (padrão: todos)")
    p.add_argument("--max-texts", type=int, default=None,
                   help="Limitar número de frases")
    p.add_argument("--save-audio", action="store_true",
                   help="Salvar áudios gerados em eval/audio/")
    p.add_argument("--output-dir", type=Path, default=Path("eval/results"),
                   help="Diretório de saída dos relatórios")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--tts-voice", default="pt-BR-FranciscaNeural")

    # MLflow
    ml = p.add_argument_group("MLflow (opcional — requer: pip install mlflow)")
    ml.add_argument("--mlflow-uri", default="",
                    help="URI do servidor MLflow (ex.: http://localhost:5000 ou ./mlruns)")
    ml.add_argument("--mlflow-experiment", default="",
                    help="Nome do experimento MLflow (padrão: tdvx-synthetic-eval)")
    ml.add_argument("--run-name", default="",
                    help="Nome do run MLflow (padrão: eval-<timestamp>)")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Textos ────────────────────────────────────────────────────────────────
    if args.texts_file and args.texts_file.exists():
        texts = [l.strip() for l in args.texts_file.read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.startswith("#")]
        log.info("Carregadas %d frases de %s", len(texts), args.texts_file)
    else:
        texts = DEFAULT_TEXTS
        log.info("Usando %d frases padrão pt-BR", len(texts))

    if args.max_texts:
        texts = texts[: args.max_texts]

    # ── Modelo ────────────────────────────────────────────────────────────────
    import torch

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    compute_type = "int8_float16" if device == "cuda" else "int8"
    log.info("Carregando modelo '%s' em %s (%s)…", args.model_path, device, compute_type)

    model = WhisperModel(args.model_path, device=device, compute_type=compute_type)
    log.info("Modelo carregado.")

    audio_dir = Path("eval/audio") if args.save_audio else None
    if audio_dir:
        audio_dir.mkdir(parents=True, exist_ok=True)

    # ── MLflow ────────────────────────────────────────────────────────────────
    with MlflowTracker(args) as tracker:
        tracker.log_params(args, len(texts))
        tracker.set_tags(args.model_path, device)

        # ── Avaliação ─────────────────────────────────────────────────────────
        results: List[EvalResult] = []
        total = len(texts) * len(args.noise_types)
        done = 0

        for tid, text in enumerate(texts):
            log.info('[%d/%d] TTS: "%s"', tid + 1, len(texts), text[:50])

            try:
                clean_audio = generate_clean_audio(text, args.tts_voice)
            except Exception as e:
                log.error("Erro ao gerar TTS: %s", e)
                continue

            if audio_dir:
                sf.write(audio_dir / f"{tid:03d}_clean.wav", clean_audio, SAMPLE_RATE)

            for noise_type in args.noise_types:
                done += 1
                log.info("  [%d/%d] Ruído: %s", done, total, noise_type)

                try:
                    noisy_audio = apply_noise(clean_audio, noise_type)
                except Exception as e:
                    log.warning("  Erro ao aplicar ruído '%s': %s", noise_type, e)
                    continue

                if audio_dir:
                    sf.write(audio_dir / f"{tid:03d}_{noise_type}.wav", noisy_audio, SAMPLE_RATE)

                t0 = time.perf_counter()
                try:
                    transcription, lang, avg_log_prob, no_speech_prob = transcribe(model, noisy_audio)
                except Exception as e:
                    log.warning("  Erro na transcrição: %s", e)
                    transcription, lang, avg_log_prob, no_speech_prob = "", "?", -99.0, 1.0

                elapsed = time.perf_counter() - t0
                wer_score = _safe_wer(text, transcription)

                result = EvalResult(
                    text_id=tid,
                    original_text=text,
                    noise_type=noise_type,
                    transcription=transcription,
                    wer=wer_score,
                    inference_time_s=round(elapsed, 3),
                    language_detected=lang,
                    avg_log_prob=round(avg_log_prob, 4),
                    no_speech_prob=round(float(no_speech_prob), 4),
                )
                results.append(result)

            print_table([r for r in results if r.text_id == tid])

        # ── Relatórios e métricas MLflow ──────────────────────────────────────
        if results:
            json_path, csv_path = save_reports(results, args.output_dir)
            tracker.log_metrics(results)
            tracker.log_artifacts(json_path, csv_path)

        log.info("Avaliação concluída. %d resultados gerados.", len(results))


if __name__ == "__main__":
    main()
