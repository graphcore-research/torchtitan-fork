#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null && pwd)

TMP_DIR=${HOME}/alexandrep-tmp
VIRTUAL_ENV_DIR="${TMP_DIR}/.venv${1}"
mkdir -p $TMP_DIR

if [ ! -d "${VIRTUAL_ENV_DIR}" ]; then
  virtualenv -p python3 "${VIRTUAL_ENV_DIR}"
else
    echo "Virtual environment already exists"
fi

source "${VIRTUAL_ENV_DIR}/bin/activate"

# alias  git config user.name "Alexandre Payot" && git config user.email "alexandrep@graphcore.ai"