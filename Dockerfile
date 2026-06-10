# Worker GPU do RunPod — OmniVoice (k2-fsa), Apache 2.0, 646 idiomas, RTF 0.025.
# NOTA: confira a tag da imagem base no RunPod (atualizam). Qualquer base
# pytorch+cuda recente serve. Se 'omnivoice' puxar um torch conflitante,
# fixe a versao do torch da base aqui.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

ARG MODEL_ID=k2-fsa/OmniVoice
ENV MODEL_ID=${MODEL_ID} \
    HF_HOME=/app/hf \
    DTYPE=fp16 \
    CHUNK_GAP_MS=0

# FIX: a base do RunPod ships com torch 2.4.0, mas o `transformers` recente
# (puxado pelo omnivoice) precisa de torch >= 2.6 (usa torch.float8_e8m0fnu).
# Sem esse upgrade, o handler crasha com AttributeError no import. Pinamos
# torch primeiro com wheels cu124 (mesma CUDA da base) — pip mantem esse torch
# quando resolver as deps do omnivoice depois.
RUN pip install --no-cache-dir --upgrade \
      --index-url https://download.pytorch.org/whl/cu124 \
      "torch>=2.6.0"

RUN pip install --no-cache-dir runpod omnivoice soundfile numpy huggingface_hub

# IMPORTANTE: baixa os pesos NA IMAGEM (build time) pra o cold start nao baixar
# o modelo a cada subida de worker. E isso que deixa o boot rapido.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_ID}')"

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
