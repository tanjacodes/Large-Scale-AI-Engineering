#!/usr/bin/env python3
import os
import socket
import torch
import torch.distributed as dist
import time
import matplotlib.pyplot as plt
import numpy as np

# Read environment variables set by torchrun
rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

# Initializes the default (global) process group
dist.init_process_group(backend="nccl")

# parameters
n_warmup = 5
n_measurements = 25

# Limit GPU allocation of this process to only one GPU
torch.cuda.set_device(local_rank)

N = 2 ** 30  # ~1.1 billion elements
tensor = torch.full((N,), fill_value=rank, dtype=torch.float32, device="cuda")

torch.cuda.synchronize()

measurement_number = 0

# Warmup
for _ in range(n_warmup):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    dist.barrier()

measured_throughput_global = torch.zeros((n_measurements, 23), device="cuda")

for i in range(10,33):
    N = 2 ** i 
    for j in range(n_measurements):
        tensor = torch.full((N,), fill_value=rank, dtype=torch.float32, device="cuda")

        # Force a CUDA synchronization point before measuring time
        torch.cuda.synchronize()
        dist.barrier()

        # Record the start time
        start = time.time()

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        # Force a CUDA synchronization again to ensure the operation is completed before measuring the end time.
        torch.cuda.synchronize()
        dist.barrier()

        # Measure end time
        end = time.time()
        elapsed_seconds = end - start

        total_bytes = tensor.nelement() * 4  # convert elements to bytes
        total_gbs = total_bytes / (1024**3)  # convert to GB
        throughput = total_gbs / elapsed_seconds  # GB/s

        measured_throughput_global[j, i-10] = throughput

        if rank == 0:
            print(f"Measurement {measurement_number}: Tensor Size: 2^{i} ({N} elements), Throughput: {throughput:.2f} GB/s")
        measurement_number += 1


nodes_groups = [
    dist.new_group([0,1,2,3]),
    dist.new_group([4,5,6,7]),
    dist.new_group([8,9,10,11]),
    dist.new_group([12,13,14,15]),
]

measured_throughput_intra_node = torch.zeros((n_measurements, 23), device="cuda")

for i in range(10,33):
    N = 2 ** i 
    tensor = torch.full((N,), fill_value=rank, dtype=torch.float32, device="cuda")
    for _ in range(n_warmup):
        # Executes the reduce op on the group to which this process belongs.
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=nodes_groups[rank // 4])
        torch.cuda.synchronize()
        dist.barrier()

    for j in range(n_measurements):
        tensor = torch.full((N,), fill_value=rank, dtype=torch.float32, device="cuda")

        torch.cuda.synchronize()
        dist.barrier(group=nodes_groups[rank // 4])
        start = time.time()

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=nodes_groups[rank // 4])

        dist.barrier(group=nodes_groups[rank // 4])

        end = time.time()

        elapsed_seconds_all_reduce = end - start

        total_bytes = tensor.numel() * tensor.element_size()  # convert elements to bytes
        total_gbs = total_bytes / (1024**3)  # convert to GB
        throughput = total_gbs / elapsed_seconds_all_reduce


        measured_throughput_intra_node[j,i-10] = throughput

        if rank == 0:
            print(f"Measurement {measurement_number}: Tensor Size: 2^{i} ({N} elements), Throughput: {throughput:.2f} GB/s")
        measurement_number += 1


inter_node_groups = [
    dist.new_group([0,4,8,12]),
    dist.new_group([1,5,9,13]),
    dist.new_group([2,6,10,14]),
    dist.new_group([3,7,11,15]),
]

measured_throughput_inter_node = torch.zeros((n_measurements, 23), device="cuda")

for _ in range(n_warmup):
    tensor = torch.full((N,), fill_value=rank, dtype=torch.float32, device="cuda")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=inter_node_groups[rank % 4])
    torch.cuda.synchronize()
    dist.barrier(group=inter_node_groups[rank % 4])

for i in range(10,33):
    N = 2 ** i
    for j in range(n_measurements):
        tensor = torch.full((N,), fill_value=rank, dtype=torch.float32, device="cuda")

        torch.cuda.synchronize()
        dist.barrier(group=inter_node_groups[rank % 4])
        start = time.time()

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=inter_node_groups[rank % 4])

        torch.cuda.synchronize()
        dist.barrier(group=inter_node_groups[rank % 4])
        end = time.time()

        elapsed_seconds_all_reduce = end - start

        total_bytes = tensor.numel() * tensor.element_size()  # convert elements to bytes
        total_gbs = total_bytes / (1024**3)  # convert to GB
        throughput = total_gbs / elapsed_seconds_all_reduce


        measured_throughput_inter_node[j,i-10] = throughput

        if rank == 0:
            print(f"Measurement {measurement_number}: Tensor Size: 2^{i} ({N} elements), Throughput: {throughput:.2f} GB/s")
        measurement_number += 1

# Synchronize across ranks
dist.barrier()
dist.all_reduce(measured_throughput_global, op=dist.ReduceOp.SUM)
dist.all_reduce(measured_throughput_intra_node, op=dist.ReduceOp.SUM)
dist.all_reduce(measured_throughput_inter_node, op=dist.ReduceOp.SUM)
dist.barrier()

if rank == 0:
    measured_throughput_global = measured_throughput_global.cpu().numpy() / world_size
    measured_throughput_intra_node = measured_throughput_intra_node.cpu().numpy() / world_size
    measured_throughput_inter_node = measured_throughput_inter_node.cpu().numpy() / world_size

    x = np.arange(10, 33)
    x_labels = [f'$2^{{{i}}}$' for i in x]

    plt.figure(figsize=(12, 6))

    # === BOX PLOTS ===
    positions = np.arange(len(x))
    width = 0.25  # small offset for multiple boxplots per x
    offset = [-width, 0, width]

    data_series = [
        (measured_throughput_global, 'Global All-Reduce', '#1f77b4'),
        (measured_throughput_intra_node, 'Intra-Node All-Reduce', '#2ca02c'),
        (measured_throughput_inter_node, 'Inter-Node All-Reduce', '#d62728')
    ]

    for i, (data, label, color) in enumerate(data_series):
        bp = plt.boxplot(
            data,  # shape (num_sizes, num_runs)
            positions=positions, #+ offset[i],
            widths=0.2,
            patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.2),
            medianprops=dict(color=color, linewidth=2),
            whiskerprops=dict(color=color, linewidth=1),
            capprops=dict(color=color, linewidth=1),
            flierprops=dict(marker='.', color=color, alpha=0.5)
        )

        # === MEAN POINTS ===
        means = data.mean(axis=0)
        stds = data.std(axis=0)

        plt.plot(
		#+ offsets
            positions , means, 'o-', color=color, label=label
        )


    # === AXES SETTINGS ===
    plt.yscale('log')
    plt.xlabel('Tensor Size (N)')
    plt.ylabel('Throughput (GB/s)')
    plt.title('All-Reduce Throughput Comparison (mean ± std)')
    plt.xticks(positions, x_labels)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    plt.savefig('/iopsstor/scratch/cscs/tsrindran/all_reduce_throughput_2.pdf')
    plt.close()


dist.destroy_process_group()
