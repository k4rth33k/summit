# syntax=docker/dockerfile:1
# Build from the Summit repository root. Compilation uses CPU only; no --gpus.
FROM python:3.12-slim-bookworm AS runtime-check
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential && rm -rf /var/lib/apt/lists/*
COPY summit/runtime-requirements.txt /tmp/runtime-requirements.txt
RUN pip install --no-cache-dir -r /tmp/runtime-requirements.txt
RUN pip install --no-deps 'NexRL @ git+https://github.com/nex-agi/NexRL.git@6860191994413876a9d7f1431d79c29d114a12e6'
RUN printf '%s\n' '{"phantora":null,"torch":null,"nexrl":"6860191994413876a9d7f1431d79c29d114a12e6"}' > /opt/phantora-provenance.json
WORKDIR /tmp

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS torch-build
ARG MAX_JOBS=4
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git ca-certificates curl build-essential cmake ninja-build ccache pkg-config \
    python3.12-dev python3.12-venv libopenmpi-dev libssl-dev libclang-dev \
    libpng-dev libjpeg-dev && rm -rf /var/lib/apt/lists/*
RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:/usr/local/cargo/bin:$PATH
ENV CUDA_HOME=/usr/local/cuda
ENV PYTORCH_BUILD_VERSION=2.7.1 PYTORCH_BUILD_NUMBER=1
ENV BUILD_TEST=0
ENV TORCH_CUDA_ARCH_LIST=8.0 USE_CUDNN=0 USE_CUSPARSELT=0
ENV CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache
ENV CCACHE_DIR=/var/cache/ccache
WORKDIR /phantora
RUN git init pytorch && cd pytorch && git remote add origin https://github.com/QDelta/pytorch.git && \
    git fetch --depth 1 origin 43d1bb713f140464e7d6ae7014b1de0f140dc94b && git checkout FETCH_HEAD && \
    git submodule update --init --recursive --depth 1
WORKDIR /phantora/pytorch
RUN pip install --no-cache-dir -r requirements.txt 'numpy==1.26.4'
# Optional late override lets builders adjust parallelism without invalidating
# source/dependency layers. MAX_JOBS remains the normal user-facing setting.
ARG TORCH_COMPILE_JOBS
RUN --mount=type=cache,target=/var/cache/ccache \
    MAX_JOBS=${TORCH_COMPILE_JOBS:-${MAX_JOBS}} pip install --no-build-isolation --no-deps --verbose -e .

FROM torch-build AS simulator
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --no-modify-path --profile minimal --default-toolchain 1.90.0 && \
    cargo install --locked cbindgen --version 0.29.2
WORKDIR /phantora/upstream
RUN git init && git remote add origin https://github.com/QDelta/Phantora.git && \
    git fetch --depth 1 origin 301d9d976a13067cc0f5438912395a11efba5f8e && git checkout FETCH_HEAD
COPY simulation/patch_upstream.py /tmp/patch_upstream.py
RUN python /tmp/patch_upstream.py /phantora/upstream
ENV LIBTORCH=/phantora/pytorch/torch
ENV LD_LIBRARY_PATH=/phantora/pytorch/torch/lib:/usr/local/cuda/lib64
RUN mkdir -p build && CARGO_BUILD_JOBS=4 make dist && cp -a dist /phantora/dist

FROM simulator AS runtime
COPY summit/runtime-requirements.txt /tmp/runtime-requirements.txt
# The already-installed patched torch satisfies the pin. Never replace it with
# a stock wheel: runtime validation also checks the function-tracer entry point.
RUN pip install --no-cache-dir -r /tmp/runtime-requirements.txt
RUN pip install --no-deps 'NexRL @ git+https://github.com/nex-agi/NexRL.git@6860191994413876a9d7f1431d79c29d114a12e6'
RUN python -c 'import torch; assert torch.__version__.split("+")[0] == "2.7.1"; assert hasattr(torch.profiler, "enable_function_tracer")'
RUN printf '%s\n' '{"phantora":"301d9d976a13067cc0f5438912395a11efba5f8e","torch":"43d1bb713f140464e7d6ae7014b1de0f140dc94b","nexrl":"6860191994413876a9d7f1431d79c29d114a12e6"}' > /opt/phantora-provenance.json
WORKDIR /tmp
ENTRYPOINT ["python", "-m", "summit.simulation.container"]
