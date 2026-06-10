# FIX #4: troca a base do RunPod (torch 2.4) pela imagem OFICIAL do PyTorch.
# Motivo: as 3 tentativas anteriores de upgrade torch via pip ainda deixavam
# torchaudio com versao errada (cu121 no lugar de cu124), causando CUDA
# mismatch em runtime. A imagem oficial pytorch/pytorch ja vem com torch +
# torchaudio + cuDNN todos cu124, matching garantido. Eliminamos o pip dance.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

ARG MODEL_ID=k2-fsa/OmniVoice
ENV MODEL_ID=${MODEL_ID} \
    HF_HOME=/app/hf \
    DTYPE=fp16 \
    CHUNK_GAP_MS=0 \
    DEBIAN_FRONTEND=noninteractive

# System libs que omnivoice/soundfile precisam (a imagem oficial e mais "slim"
# que a do RunPod, entao tem que adicionar manualmente).
RUN apt-get update && apt-get install -y --no-install-recommends \
      libsndfile1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Instala omnivoice + runpod + deps. NAO mexe em torch/torchaudio (ja vem certo).
# Se omnivoice tentar reinstalar torch, --upgrade-strategy=only-if-needed protege.
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed \
      runpod omnivoice soundfile numpy huggingface_hub

# Baixa pesos do OmniVoice na imagem pra cold start nao baixar 5GB toda vez.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_ID}')"

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
