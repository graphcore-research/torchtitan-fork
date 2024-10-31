# Low bits (pre) LLMs training

## Setup

```bash
export PYTHONPATH="${PYTHONPATH}:torchtitan/"
pip install -e ./
pip install -r ./torchtitan/requirements.txt
```

## Testing the install on a single accelerator

Using C4 test dataset and the debug model:
```bash
NGPU=1 ./run_llama_train.sh
```

## Training

MX numerics training experiments...

## Development
