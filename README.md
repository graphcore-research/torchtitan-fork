# Low bits LLMs (pre-)training

## Setup

If required, setup a new Python environment:
```bash
python3 -m venv env
source env/bin/activate
```

Get submodules
```
git submodule update --init --recursive
```

Editable install of `low-bits-training` package, and `torchtitan` requirements:
```bash
pip install -e ./
pip install -r ./torchtitan/requirements.txt
```

Following `torchtitan/README.md` (See "Installation"), you may need to install the latest nightly PyTorch:
```bash
pip3 install --force-reinstall --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124 # or cu118, cu121
```

And you will need to get a tokeniser (again see `torchtitan/README.md`, "Downloading a tokenizer").

### Testing the install on a single/multiple accelerator(s)

Using C4 test dataset and the debug model:
```bash
WANDB_MODE=disabled NGPU=1 ./run_llama_train.sh  # or NGPU=8
```
Please see the [Troubleshooting](#troubleshooting) section below in case this script does not run properly.

## Training

## Locally, for 1-8 GPUs setup

For training directly on a server with 1-8 GPUs, use the following script (directly calling `torchrun`):
```bash
CONFIG_FILE=xxx.toml NGPU=8 ./run_llama_train.sh
```

## GPU cluster, using `slurm`

For training on GPU cluster, you will need to `git clone` the repository in a network filesystem shared between the compute nodes (e.g. `/data` on LambdaLabs). Then, setup a couple of environnment variable inside a `.env` file:
```bash
export HF_TOKEN=hf_         # HF token for getting the tokenizer
export WANDB_API_KEY=...    # WANDB key for experiments logging
export GC_USER=alexandrep   # Graphcore username for virtual env
```

Then you launch the training with:
```bash
WANDB_PROJECT=some-other-project CONFIG_FILE=xxx.toml bash submit.sh
```


## Troubleshooting

* `segfault` on H100 LambdaLabs instances. Update `nvidia-nccl-cu12` to latest: `pip install -U nvidia-nccl-cu12`. Note that `pip` will be warning that it is not compatible with `torch`.
* `[rank0]:wandb.errors.errors.UsageError: api_key not configured`. Setup Weights & Biases with the command line `wandb login` (requires an API key).
* `torch.distributed` import error. Outdated version of PyTorch. Make sure you set up your Python virtual env on the local SSD when using LambdaLabs.

### Resources

* Guide on debugging GPU networking: https://github.com/stas00/ml-engineering/tree/master/network/debug
* Pytorch distributed backend NCCL environment variables: https://pytorch.org/docs/stable/distributed.html#other-nccl-environment-variables

### Debugging `torchrun`

#### Debug python code

Add this to your VS Code `launch.json`:

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: Module",
            "type": "debugpy",
            "request": "launch",
            "module": "torch.distributed.launch",
            "env": {"TORCH_LOGS": "+all"},
            "args": [
                "--nproc_per_node=1",
                "--use-env",
                "--rdzv_backend=c10d",
                "--rdzv_endpoint=localhost:0",
                "--local-ranks-filter=0",
                "--role=rank",
                "--tee=3",
                "train.py",
                "--job.config_file",
                "./train_configs/debug_model.toml"
            ],
            "cwd": "${workspaceFolder}"
        }
    ]
}
```

### GDB with distributed torch (find where a segfault is)

On the command line, first install the python debug symbols with `sudo apt install gdb python3-dbg`

```console
$ gdb --args python -m torch.distributed.launch --nproc_per_node=1 --use-env --rdzv_backend c10d --rdzv_endpoint=localhost:0 --local-ranks-filter 0 --role rank --tee 3 train.py --job.config_file ./train_configs/debug_model.toml
set detach-on-fork off      # gdb stays attached to subprocesses (but only 1 at a time runs)
set follow-fork-mode child      # Follow the child
run
```

Then execute individual programs and threads (inferiors) as needed to reach the segfault
see [Documentation on inferiors and programs](https://www.zeuthen.desy.de/unix/unixguide/infohtml/gdb/Inferiors-and-Programs.html).
Useful GDB commands:

* `info inferior`: list inferiors
* `inferior N`: move to that inferior to inspect/execute it.

## Local development

```bash
pre-commit install
```
