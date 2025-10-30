#!/usr/bin/env python3
import os
import socket
import torch
import torch.distributed as dist
N = 2 ** 30  # ~1.1 billion elements
tensor = torch.full((N,), fill_value=rank, dtype=torch.float32, device="cuda")
# Read environment variables set by torchrun
rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])

# Initializes the default (global) process group
dist.init_process_group(backend="nccl")




# Limit GPU allocation of this process to only one GPU
torch.cuda.set_device(local_rank)

# Create a float32 tensor on each rank with a single element of value 'rank' and move it to the GPU.
local_tensor = torch.tensor([rank], dtype=torch.float32).cuda()
print(f"[Python] rank={rank} | local_tensor={local_tensor.item()}")

# Warmup
for _ in range(5):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# Force a CUDA synchronization point before measuring time
torch.cuda.synchronize()

# Record the start time
start = time.time()

dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# Force a CUDA synchronization again to ensure the operation is completed before measuring the end time.
torch.cuda.synchronize()

# Measure end time
end = time.time()
elapsed_seconds = end - start

#Check/Assert
expected_val = world_size*(world_size-1)/2
assert torch.allclose(
    tensor,
    torch.full_like(tensor, expected_val)
), f"[Python] rank={rank} | all-Reduce mismatch: expected {expected_val}, got {tensor[0].item()} in first element."

#Report throughput
total_bytes = tensor.nelement() * 4  # convert elements to bytes
total_gbs = total_bytes / (1024**3)  # convert to GB
throughput = total_gbs / elapsed_seconds  # GB/s

print(f"[Python] rank={rank} | transferred {total_gbs:.2}GB | throughput={throughput:.4}GB/s")
# Cleanup
dist.destroy_process_group()
