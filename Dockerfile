# Judges: dataset read-only at /data, write /out.
#   docker build -t ylookup .
#   docker run --rm \
#     -e TINKER_API_KEY \
#     -e TINKER_PROJECT_ID \
#     -v /path/to/spreadsheetbench_verified_400:/data:ro \
#     -v /path/to/out:/out \
#     ylookup
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_DEV=1
COPY research/pyproject.toml research/uv.lock ./
RUN uv sync --frozen --extra tinker --no-install-project
COPY research/baseline ./baseline
COPY research/sb.py ./
ENTRYPOINT ["uv", "run", "--frozen", "--extra", "tinker", "baseline/tinker_predict.py"]
CMD ["--dataset-dir", "/data", "--out-dir", "/out", "--base-model", "Qwen/Qwen3.8-27B", "--renderer", "qwen3_8_xhigh_reasoning", "--agent", "auto", "--model-path", "tinker://600ccd82-58a6-5eee-b276-7c212844dd0d:train:0/sampler_weights/final"]
