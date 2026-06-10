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

# FIX #2: combina torch+torchaudio+torchvision+omnivoice numa UNICA pip install
# pra resolver todas as deps de uma vez. O fix anterior (so torch upgrade) gerou
# CUDA mismatch (PyTorch cu121 vs TorchAudio cu124) porque o segundo
# `pip install omnivoice` puxou torch novamente do PyPI default. Aqui:
#   --index-url cu124       -> primario (torch/torchaudio/torchvision com CUDA 12.4)
#   --extra-index-url PyPI  -> fallback (omnivoice/runpod/soundfile/etc)
# Resultado: torch e torchaudio na mesma CUDA, sem mismatch no boot.
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu124 \
      --extra-index-url https://pypi.org/simple \
      "torch>=2.6.0" "torchaudio>=2.6.0" "torchvision>=0.21.0" \
      runpod omnivoice soundfile numpy huggingface_hub

# IMPORTANTE: baixa os pesos NA IMAGEM (build time) pra o cold start nao baixar
# o modelo a cada subida de worker. E isso que deixa o boot rapido.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_ID}')"

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
