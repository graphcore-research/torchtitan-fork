# Low bits LLMs (pre-)training

## Setup

If required, setup a new Python environment:
```bash
python3 -m venv env
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
WANDB_MODE=disabled NGPU=1 ./run_llama_train.sh  # or NGPU=8
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
