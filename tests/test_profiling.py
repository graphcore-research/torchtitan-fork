# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
import torch
from torch.profiler import profile
import torch.profiler
from pathlib import Path
import json
from low_bits_training.profiling import get_trace_handler

import pytest


@pytest.fixture()
def initialise_torch_distributed(monkeypatch):
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("MASTER_PORT", "8001")
    torch.distributed.init_process_group("gloo", world_size=1)
    yield
    torch.distributed.destroy_process_group()


def check_flops_in_trace(trace_file):
    """Check that the trace contains flops counts in the events"""
    trace_contents = json.loads(trace_file.read_text())
    flops_count = 0
    for trace_event in trace_contents["traceEvents"]:
        flops_count += trace_event.get("args", {}).get("flops", 0)
    assert flops_count > 0, "no flops were found in the trace"


@pytest.mark.parametrize("mem_timeline", [None, "json", "raw.json.gz"])
@pytest.mark.parametrize("with_flops", [True, False])
def test_profile_handler(
    initialise_torch_distributed, tmp_path, mem_timeline, with_flops
):
    rank = 1
    with_mem = bool(mem_timeline)
    trace_handler = get_trace_handler(tmp_path, mem_timeline, rank, with_flops, "cpu")
    with profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
        ],
        on_trace_ready=trace_handler,
        record_shapes=True,
        with_flops=with_flops,
        profile_memory=with_mem,
        with_stack=with_mem,
    ) as profile_with_flops:
        x = torch.randn([100, 100], device="cpu")
        y = torch.randn([100, 100], device="cpu")
        _ = x.mm(y)
        profile_with_flops.step()

    trace_dir = Path(tmp_path) / "iteration_1"
    assert (
        trace_dir.exists()
    ), f"The directory in which traces are expected did not exist: {trace_dir}"
    trace_files = list(trace_dir.glob(f"rank{rank}.*.pt.trace.json"))
    trace_dir_and_contents = f"{trace_dir} with contents: {list(trace_dir.iterdir())}"
    assert len(trace_files) == 1, f"Trace file not found in {trace_dir_and_contents}"
    trace_file = trace_files[0]
    if mem_timeline:
        assert (
            len(list(trace_dir.glob(f"*memory*{mem_timeline}"))) == 1
        ), f"Memory files not found in {trace_dir_and_contents}"
    if with_flops:
        check_flops_in_trace(trace_file)
