"""RunPod serverless worker — Qwen3-TTS voice cloning.

Roda na GPU do RunPod. Recebe um "shard" (bloco de chunks de texto) + a amostra
de voz, gera o audio de cada chunk clonando a voz, e devolve o audio do shard
concatenado + as duracoes por chunk (pro coordenador montar o SRT) + metricas.

O modelo carrega UMA vez no cold start e fica quente. O prompt de clonagem da
voz e cacheado por amostra, entao so e calculado na primeira requisicao.
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
from qwen_tts import Qwen3TTSModel

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
# "sdpa" funciona out-of-the-box. "flash_attention_2" e mais rapido SE flash-attn
# estiver instalado na imagem (mais chato de buildar). Pro PoC, sdpa.
ATTN = os.environ.get("ATTN_IMPL", "sdpa")
# Silencio inserido entre chunks (ms). 0 = SRT alinha exato. >0 = costura mais natural.
GAP_MS = int(os.environ.get("CHUNK_GAP_MS", "0"))

print(f"[boot] carregando {MODEL_ID} (attn={ATTN}) ...", flush=True)
_t0 = time.time()
model = Qwen3TTSModel.from_pretrained(
    MODEL_ID,
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation=ATTN,
)
print(f"[boot] modelo carregado em {time.time() - _t0:.1f}s", flush=True)

_prompt_cache = {}


def _get_clone_prompt(ref_audio_b64, ref_text):
    key = hashlib.sha256((ref_audio_b64[:1024] + "|" + ref_text).encode()).hexdigest()
    cached = _prompt_cache.get(key)
    if cached is not None:
        return cached
    raw = base64.b64decode(ref_audio_b64)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(raw)
        ref_path = f.name
    prompt = model.create_voice_clone_prompt(ref_audio=ref_path, ref_text=ref_text)
    _prompt_cache[key] = prompt
    return prompt


def handler(job):
    inp = job["input"]
    texts = inp["texts"]                       # list[str] — bloco de chunks
    language = inp.get("language", "Portuguese")
    ref_audio_b64 = inp["ref_audio_b64"]
    ref_text = inp["ref_text"]

    prompt = _get_clone_prompt(ref_audio_b64, ref_text)

    sr = None
    pieces = []
    durations = []
    t0 = time.time()
    for txt in texts:
        wavs, sr = model.generate_voice_clone(
            text=txt, language=language, voice_clone_prompt=prompt
        )
        arr = np.asarray(wavs, dtype=np.float32).reshape(-1)
        pieces.append(arr)
        durations.append(round(len(arr) / sr, 3))
        if GAP_MS > 0:
            pieces.append(np.zeros(int(sr * GAP_MS / 1000), dtype=np.float32))
    gen_seconds = round(time.time() - t0, 3)

    audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr or 24000, format="WAV", subtype="PCM_16")
    audio_seconds = round(len(audio) / sr, 3) if sr else 0.0

    return {
        "audio_b64": base64.b64encode(buf.getvalue()).decode(),
        "sample_rate": sr,
        "n_chunks": len(texts),
        "chunk_durations": durations,
        "gen_seconds": gen_seconds,
        "audio_seconds": audio_seconds,
        "rtf": round(gen_seconds / audio_seconds, 4) if audio_seconds else None,
    }


runpod.serverless.start({"handler": handler})
