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

# FIX #3: pin EXATO com sufixo +cu124. As wheels do pytorch.org/whl/cu124
# tem versao tipo "2.6.0+cu124" (PEP 440 local version). PyPI tem so "2.6.0"
# (sem sufixo = build cu121 default). Pinando "==2.6.0+cu124" forca pip a usar
# SO a wheel cu124, sem chance de pegar cu121 do PyPI por engano.
#
# Hist do bug: fix #2 usou ">=2.6.0" -> pip podia escolher uma versao mais
# nova do PyPI (cu121) ignorando o index-url cu124. Resultado: CUDA mismatch
# em runtime, auto-rollback do RunPod.
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu124 \
      --extra-index-url https://pypi.org/simple \
      "torch==2.6.0+cu124" "torchaudio==2.6.0+cu124" "torchvision==0.21.0+cu124" \
      runpod omnivoice soundfile numpy huggingface_hub

# IMPORTANTE: baixa os pesos NA IMAGEM (build time) pra o cold start nao baixar
# o modelo a cada subida de worker. E isso que deixa o boot rapido.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_ID}')"

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
