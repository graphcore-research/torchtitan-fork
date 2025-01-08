#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import datasets
import wandb
import os
import subprocess
import sys
import time
import argparse
import re


def sync_between_nodes(source_path, source_host, dest_path, dest_host):
    """Synchronise a folder from one node to another using scp"""
    # Construct the synchronisation command
    rsync_command = ["scp", "-r", f"{source_path}", f"{dest_host}:{dest_path}"]
    print(rsync_command)
    try:
        # Run the command and capture output
        process = subprocess.Popen(
            rsync_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        # Print output in real-time
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                print(output.strip())

        # Get the return code
        return_code = process.poll()

        # Check if there were any errors
        if return_code != 0:
            _, error = process.communicate()
            print(f"Error occurred: {error}", file=sys.stderr)
            return False

        return True

    except Exception as e:
        print(f"An error occurred: {str(e)}", file=sys.stderr)
        return False


def main(args):
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name=f"dataset-download-{args.hf_dataset}",
    )
    finished = False
    max_retry = 100
    retries = 0
    while not finished:
        try:
            datasets.load_dataset(args.hf_dataset, name="default", split="train")
            finished = True
        except Exception as e:
            print(f"Error while downloading: {e}")
            time.sleep(60)
            retries += 1
            finished = retries >= max_retry

    print("Finished downloading starting to copy folder")
    # Define the paths and hosts
    # This will work but we should use an attribute from the dataset object
    cache_path_name = re.sub(
        "[/]", "_", re.sub("([A-Z])", r"_\g<1>", args.hf_dataset).lower()
    )
    source_path = f"/home/ubuntu/.cache/huggingface/datasets/{cache_path_name}"
    source_host = os.environ["head_node"]

    dest_path = "/home/ubuntu/.cache/huggingface/datasets/"
    if source_host == "graphcore-test-cluster-node-001":
        dest_host = "graphcore-test-cluster-node-002"
    else:
        dest_host = "graphcore-test-cluster-node-001"

    # Run the rsync command
    success = sync_between_nodes(source_path, source_host, dest_path, dest_host)

    if success:
        print("Rsync completed successfully!")
    else:
        print("Rsync failed!", file=sys.stderr)

    run.finish()


def get_parser():
    parser = argparse.ArgumentParser(description="Download and copy dataset")

    parser.add_argument(
        "--hf_dataset",
        type=str,
        default="cerebras/SlimPajama-627B",
        help="Hugging Face dataset to download",
    )
    parser.add_argument("--entity", type=str, default="graphcore", help="Wandb entity")

    parser.add_argument(
        "--project",
        type=str,
        default="low-bits-training-dataset-download",
        help="Wandb project",
    )
    return parser


if __name__ == "__main__":
    main(get_parser().parse_args())
