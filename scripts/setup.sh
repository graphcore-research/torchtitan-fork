#!/bin/bash

# Installing `uv` for Python env management.
if ! [ -x "$(command -v uv)" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Read local env. (GC USER, ...) if existing.
if [ -f .env ]; then
  source .env
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null && pwd)

# TODO: a bit of a cleaner solution for a volatile directory.
TMP_DIR=${HOME}/${GC_USER}-tmp
VIRTUAL_ENV_DIR="${TMP_DIR}/.venv${1}"
mkdir -p $TMP_DIR

if [ ! -d "${VIRTUAL_ENV_DIR}" ]; then
  virtualenv -p python3 "${VIRTUAL_ENV_DIR}"
else
    echo "Virtual environment already exists"
fi

source "${VIRTUAL_ENV_DIR}/bin/activate"

# alias  git config user.name "GC NAME" && git config user.email "gc-name@graphcore.ai"
