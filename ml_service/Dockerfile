# ML service — the GPU/torch process (onboarding + HF inference). Build context = repo root.
#   docker build -f ml_service/Dockerfile -t engram-ml .
# The cartridge IP is a pip dependency (engram-cartridge); requirements.txt pulls it plus
# its torch/transformers deps. TODO(package): once engram-cartridge is published to our
# index this builds as-is; until then, make its wheel/sdist available to pip (a private
# index or a build context) since the sibling `cartridges/` source is not in this repo.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
WORKDIR /app
COPY ml_service/requirements.txt ml_service/requirements.txt
RUN pip install --no-cache-dir -r ml_service/requirements.txt
# The base image ships torchvision/torchaudio pinned to its torch 2.4; engram-cartridge
# bumps torch to >=2.7, orphaning them ("torchvision::nms does not exist"), which breaks
# transformers' lazy import of Qwen3. This is a text-only workload, so remove them.
RUN pip uninstall -y torchvision torchaudio
COPY ml_service ./ml_service
# Run as non-root.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8001
CMD ["uvicorn", "app:app", "--app-dir", "ml_service", "--host", "0.0.0.0", "--port", "8001"]
