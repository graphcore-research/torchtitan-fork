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

Editable install of `low-bits-training` package (which includes `torchtitan` dependencies):
```bash
pip install -e ./
# On H100, you will need to update NCCL library to avoid segfault
pip install "nvidia-nccl-cu12>=2.23.4"
```

The install process is similar with `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
uv pip install -e .
uv pip install "nvidia-nccl-cu12>=2.23.4"
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

For training on GPU cluster, you will need to `git clone ...` the repository in a network filesystem shared between the compute nodes (e.g. `/data/gc-user` on LambdaLabs). Then, setup a couple of environnment variable inside a `.env` file:
```bash
export HF_TOKEN=hf_         # HF token for getting the tokenizer
export WANDB_API_KEY=...    # WANDB key for experiments logging
export GC_USER=alexandrep   # Graphcore username for virtual env
```

The multi-node cluster training can be started with the `submit.sh` slurm script:
```bash
WANDB_PROJECT=some-other-project CONFIG_FILE=xxx.toml bash submit.sh
```
which should output
```bash
W&B project URL: 'https://wandb.ai/graphcore/some-other-project'
W&B Name: 'wb-random-name-39'
Submitted batch job 143
```

### Restarting a training job

If a training run is interrupted, you can restart it from the latest checkpoint using:
```bash
WANDB_PROJECT=some-other-project CONFIG_FILE=xxx.toml WANDB_NAME=wb-random-name-39 bash submit.sh
```
i.e. by passing the previous W&B training run name (checkpoints are saved locally in a directory with W&B training run name).

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

## Viewing profiles

When profiling is turned on in the config (see [`train_configs/llama3_8b_profiling.toml`](train_configs/llama3_8b_profiling.toml) for an example)
X types of files will be visible in the dump folder (usually in `outputs/<random>-<word>-<KK>/`):

* `profile_trace/iteration_NNNN/rank{rank}.{timestamp}.pt.trace.json` (one per rank, per profiling iteration): Chrome traces in a format compatible with [](ui.perfetto.dev) and Tensorboard.
* `profile_trace/iteration_NNNN/memory/rank{rank}.(json|raw.json.gz|json.gz)`: Pytorch memory timeline outputs (these files are hard to interpret and weakly documented - prefer using memory information from traces).
* `memory_snapshot/iteration_NNNN/rank{rank}_memory_snapshot.pickle`: Pytorch memory snapshot of the last 10000 memory allocations. You can visualize it's contents at: [](https://pytorch.org/memory_viz).

Other items:

* `comm_trace`: I've not seen this output and it will only be dumped on a NCCL error.

The recommended way to analyse the profile data is to use HolisticTraceAnalysis in a Jupyter notebook. We have a fork of the project with some helpful modifications:

```bash
uv pip install git+https://github.com/graphcore-research/HolisticTraceAnalysis-fork.git@improved-memory-analysis
```

A code sample that will do a lot of analysis and plots for you is:

```python
import hta.trace_analysis

analyzer = hta.trace_analysis.TraceAnalysis(trace_dir = str(iteration_dir))
time_spent_df = analyzer.get_temporal_breakdown()
kernel_info = analyzer.get_gpu_kernel_breakdown()
memory_events = analyzer.get_memory_timeline()
categorised_memory_timelines, memory_events = analyzer.get_memory_timeline_per_category()
# other analyses available in the  analyser object
# Also check https://github.com/graphcore-research/HolisticTraceAnalysis-fork/tree/improved-memory-analysis/examples
# and https://hta.readthedocs.io/en/latest/ for more tips
```

The profile traces can also be opened with TensorBoard in VS Code. Simply install the TensorBoard extension.

See more tips and examples on Confluence in the [1CC performance report](https://graphcore.atlassian.net/wiki/spaces/AAI/pages/4005953678/LambdaLabs+1CC+GPU+cluster+report+feedback#Profile-analysis) and the
[profiling how to guide](https://graphcore.atlassian.net/wiki/spaces/AAI/pages/4056973329/How+to+Analyse+profile+traces+generated+by+Torchtitan+WIP).

### Problems with profiling

The following limitations were identified:

* Could not get the CUPTI counters to work to get very low level information (suspected H100 specific).
