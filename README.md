# Low bits LLMs (pre-)training

## Setup

If required, setup a new Python environment:
```bash
python3 -v venv env
source env/bin/activate
```

Editable install of `low-bits-training` package, and `torchtitan` requirements:
```bash
pip install -e ./
pip install -r ./torchtitan/requirements.txt
```

Following `torchtitan` readme install, you may need to install the latest nightly PyTorch:
```bash
pip3 install --force-reinstall --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124 # or cu118, cu121
```

### Testing the install on a single/multiple accelerator(s)

Using C4 test dataset and the debug model:
```bash
NGPU=1 ./run_llama_train.sh  # or NGPU=8
```
Please see the [Troubleshooting](#troubleshooting) section below in case this script does not run properly.

## Training

MX numerics training experiments...


## Troubleshooting

* `segfault` on H100 LambdaLabs instances. Update `nvidia-nccl-cu12` to latest: `pip install -U nvidia-nccl-cu12`. Note that `pip` will be warning that it is not compatible with `torch`.
* `[rank0]:wandb.errors.errors.UsageError: api_key not configured`. Setup Weights & Biases with the command line `wandb login` (requires an API key).
* `torch.distributed` import error. Outdated version of PyTorch. Make sure you set up your Python virtual env on the local SSD when using LambdaLabs.

### Resources

* Guide on debugging GPU networking: https://github.com/stas00/ml-engineering/tree/master/network/debug

## Local development

```bash
pre-commit install
```
