# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Callable
import contextlib
import os
import time
from pathlib import Path
import json
import warnings

import torch
from .config_manager import JobConfig
from torchtitan.logging import logger
import torchtitan.profiling


# the number of warmup steps before the active step in each profiling cycle
WARMUP = 3

# how much memory allocation/free ops to record in memory snapshots
MEMORY_SNAPSHOT_MAX_ENTRIES = 100000


def _add_flops_to_trace_file(trace_file: Path, prof: torch.profiler.profile):
    """
    Add flops estimates from the profile to the trace file

    Note: this operation requires reading the trace and rewriting it. It will be slow.
    """
    events = prof.events()
    if events is None:
        return []
    flops_json = []
    flops_dict = {}
    for event in events:
        if event.flops != 0:
            flops_dict[event.id] = (event.trace_name, event.flops)
            flops_json.append(
                {"External id": event.id, "flops": event.flops, "name": event.trace_name}
            )
    if trace_file:
        trace = json.loads(trace_file.read_text())
        for event in trace["traceEvents"]:
            if args := event.get("args"):
                if name_flops := flops_dict.get(args.get("External id")):
                    assert name_flops[0] == event["name"]
                    args["flops"] = name_flops[1]
        trace_file.write_text(json.dumps(trace))
    return flops_json


def get_trace_handler(
    trace_dir, mem_timeline, rank, with_flops, device
) -> Callable[[torch.profiler.profile], None]:
    """Get the trace handler with the correct configuration to pass to the
    `on_trace_ready` argument of the Pytorch profiler."""

    # We use this getter to make the trace_handler easier to test.
    # In torchtitan it is simply nested
    def trace_handler(prof: torch.profiler.profile):
        curr_trace_dir_name = "iteration_" + str(prof.step_num)
        curr_trace_dir = os.path.join(trace_dir, curr_trace_dir_name)
        if not os.path.exists(curr_trace_dir):
            os.makedirs(curr_trace_dir, exist_ok=True)

        logger.info(f"Dumping traces at step {prof.step_num}")
        begin = time.monotonic()
        if mem_timeline:
            if "html" in str(mem_timeline):
                warnings.warn(
                    "Logging memory timeline with HTML is slow. Prefer 'json' and 'raw.json.gz' formats"
                )
            prof.export_memory_timeline(
                f"{curr_trace_dir}/rank{rank}_memory_timeline.{mem_timeline}",
                device=device,
            )
        # This simply dumps the chrome trace with a name that means tensorboard can open it
        # Tensorboard could open it anyway if you renamed - and the files generated in this
        # way can still be opened with https://ui.perfetto.dev/
        torch.profiler.tensorboard_trace_handler(
            curr_trace_dir, worker_name=f"rank{rank}"
        )(prof)
        # output flops in a way that they can be cross-referenced with the trace.
        if with_flops:
            trace_file = list(Path(curr_trace_dir).glob(f"rank{rank}.*.pt.trace.json"))[0]
            flops_json = _add_flops_to_trace_file(trace_file, prof)
            (Path(curr_trace_dir) / f"rank{rank}_flops.json").write_text(
                json.dumps(flops_json, indent=1)
            )

        logger.info(f"Finished dumping traces in {time.monotonic() - begin:.2f} seconds")
        # Profiling is a heavy operation which could cost very different amount of time
        # across all ranks. Insert a barrier to make sure all ranks have finished profiling
        # before moving on.
        # TODO: Can we find a cleaner way?
        torch.distributed.barrier()

    return trace_handler


@contextlib.contextmanager
def maybe_enable_profiling(config: JobConfig, *, global_step: int = 0):
    # get user defined profiler settings
    enable_profiling = config.profiling.enable_profiling

    if enable_profiling:
        dump_dir = config.job.dump_folder
        save_trace_dir = config.profiling.save_traces_folder
        trace_dir = os.path.join(dump_dir, save_trace_dir)
        profile_freq = config.profiling.profile_freq
        active = config.profiling.consecutive_active_steps
        mem_timeline = config.profiling.memory_timeline
        with_flops = config.profiling.with_flops_table
        turn_on_mem_timeline = bool(mem_timeline)
        rank = torch.distributed.get_rank()
        device = f"cuda:{os.getenv('LOCAL_RANK')}"
        other_kwargs = {}
        if config.profiling.experimental_cupti_stats:
            # TODO: this does not work, when activated no GPU activity is recorded.
            # This feature is experimental (broken)
            # https://hta.readthedocs.io/en/latest/source/features/cupti_counter_analysis.html#cupti-counter-analysis
            # See this issue for status: https://github.com/pytorch/pytorch/issues/125272
            # assert os.geteuid() == 0, "experimental_cupti_stats require sudo access"
            # other_kwargs = dict(
            #     experimental_config=torch.profiler._ExperimentalConfig(
            #     profiler_metrics=[
            #         "kineto__tensor_core_insts",
            #         "kineto__cuda_core_flops",
            #         "dram__bytes_read.sum",
            #         "dram__bytes_write.sum"],
            #     profiler_measure_per_kernel=True
            #     ),
            # )
            pass

        trace_handler = get_trace_handler(
            trace_dir, mem_timeline, rank, with_flops, device
        )
        logger.info(f"Profiling active. Traces will be saved at {trace_dir}")
        logger.info(f"Number of active steps: {active}")
        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir, exist_ok=True)

        warmup, active = WARMUP, active
        wait = profile_freq - (active + warmup)
        assert wait >= 0, "profile_freq must be greater than or equal to warmup + active"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=turn_on_mem_timeline,
            with_stack=turn_on_mem_timeline,
            with_flops=with_flops,  # Not in chrome traces, only on the python object
            **other_kwargs,
        ) as torch_profiler:
            torch_profiler.step_num = global_step
            yield torch_profiler
    else:
        torch_profiler = contextlib.nullcontext()
        yield None


# override
torchtitan.profiling.maybe_enable_profiling = maybe_enable_profiling
