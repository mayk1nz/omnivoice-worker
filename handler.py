"""RunPod serverless worker — OmniVoice (k2-fsa) voice cloning.

Modelo Apache 2.0, 646 idiomas (PT + BG + ...), RTF ~0.025 (40x tempo real).
Recebe um shard (bloco de chunks de texto) + amostra de voz, devolve o audio
do shard concatenado + duracoes por chunk + metricas.

A amostra de referencia e escrita uma vez no disco do worker (cache por hash)
e reusada em cada chunk pra evitar IO repetida.
"""

import base64
import hashlib
import io
import os
import tempfile
import time

import numpy as np
import soundfile as sf
import torch
import runpod
from omnivoice import OmniVoice

MODEL_ID = os.environ.get("MODEL_ID", "k2-fsa/OmniVoice")
DTYPE = torch.float16 if os.environ.get("DTYPE", "fp16") == "fp16" else torch.bfloat16
GAP_MS = int(os.environ.get("CHUNK_GAP_MS", "0"))
SR = 24000  # OmniVoice saida fixa em 24kHz

print(f"[boot] carregando {MODEL_ID} (dtype={DTYPE}) ...", flush=True)
_t0 = time.time()
model = OmniVoice.from_pretrained(MODEL_ID, device_map="cuda:0", dtype=DTYPE)
print(f"[boot] modelo carregado em {time.time() - _t0:.1f}s", flush=True)

_ref_cache = {}  # sha256(audio+text) -> path da ref no disco do worker


def _get_ref_path(ref_audio_b64, ref_text):
    key = hashlib.sha256((ref_audio_b64[:1024] + "|" + ref_text).encode()).hexdigest()
    cached = _ref_cache.get(key)
    if cached and os.path.exists(cached):
        return cached
    path = os.path.join(tempfile.gettempdir(), f"omnivoice_ref_{key[:16]}.wav")
    with open(path, "wb") as f:
        f.write(base64.b64decode(ref_audio_b64))
    _ref_cache[key] = path
    return path


def handler(job):
    inp = job["input"]
    texts = inp["texts"]                       # list[str] — bloco de chunks
    ref_audio_b64 = inp["ref_audio_b64"]
    ref_text = inp["ref_text"]

    ref_path = _get_ref_path(ref_audio_b64, ref_text)

    pieces = []
    durations = []
    t0 = time.time()
    for txt in texts:
        # OmniVoice infere o idioma do texto/referencia — sem argumento language.
        out = model.generate(text=txt, ref_audio=ref_path, ref_text=ref_text)
        arr = np.asarray(out[0], dtype=np.float32).reshape(-1)
        pieces.append(arr)
        durations.append(round(len(arr) / SR, 3))
        if GAP_MS > 0:
            pieces.append(np.zeros(int(SR * GAP_MS / 1000), dtype=np.float32))
    gen_seconds = round(time.time() - t0, 3)

    audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV", subtype="PCM_16")
    audio_seconds = round(len(audio) / SR, 3)

    return {
        "audio_b64": base64.b64encode(buf.getvalue()).decode(),
        "sample_rate": SR,
        "n_chunks": len(texts),
        "chunk_durations": durations,
        "gen_seconds": gen_seconds,
        "audio_seconds": audio_seconds,
        "rtf": round(gen_seconds / audio_seconds, 4) if audio_seconds else None,
    }


runpod.serverless.start({"handler": handler})
