#!/bin/bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Not needed with the current version of torchtitan which works with torch 2.5.1
# uv pip install --force-reinstall --prerelease allow torch --index-url https://download.pytorch.org/whl/nightly/cu124 \
# || pip install --force-reinstall --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124

uv pip install -e .
uv pip install -r ./torchtitan/requirements.txt
uv pip install 'nvidia-nccl-cu12>=2.23.4'