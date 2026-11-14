#!/bin/bash
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.uv-cache}"
export HF_HOME="${HF_HOME:-$SCRATCH/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
