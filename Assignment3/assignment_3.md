# Assignment 3: Multi-Node Runs and NCCL Communications with PyTorch

Let's learn how to use PyTorch with the NCCL backend for multi-GPU, multi-node communication. We'll look at how to initialise distributed processes in PyTorch, how to collect and distribute data across the nodes, and how to measure the time it takes to move data around.

## [1/7] Scheduling Multi-Node Runs with SLURM

Running multi-node jobs can be tricky. We will begin with an easy example. 

1. Create a new file `run_assignment_3a.sbatch` with the content below.
    ```
    #!/bin/bash
    #SBATCH --job-name=LSAIE_a3
    #SBATCH --account=large-sc-2
    #SBATCH --partition=normal
    #SBATCH --nodes=4
    #SBATCH --ntasks-per-node=4
    #SBATCH --gpus-per-node=4
    #SBATCH --time=00:03:00
    #SBATCH --output=output_3a_%j.out
    #SBATCH --error=output_3a_%j.err
    #SBATCH --environment=/capstor/store/cscs/ethz/large-sc-2/environment/ngc_pt_jan.toml

    # Stop the script if a command fails or if an undefined variable is used
    set -eo pipefail

    # The sbatch script is executed by only one node.
    echo "[sbatch-master] running on $(hostname)"

    echo "[sbatch-master] SLURM_NODELIST: $SLURM_NODELIST"
    echo "[sbatch-master] SLURM_NNODES: $SLURM_NNODES"
    echo "[sbatch-master] SLURM_NODEID: $SLURM_NODEID"

    echo "[sbatch-master] define some env vars that will be passed to the compute nodes"

    # The defined environment vars will be shared with the other compute nodes.
    export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)  
    export MASTER_PORT=12345   # Choose an unused port
    export FOOBAR=666
    export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))

    echo "[sbatch-master] execute command on compute nodes"

    # The command that will run on each process
    CMD="
    # print current environment variables
    echo \"[srun] rank=\$SLURM_PROCID host=\$(hostname) noderank=\$SLURM_NODEID localrank=\$SLURM_LOCALID wrong_host=$(hostname)\"

    # run your script which we create in the next step
    python /iopsstor/scratch/cscs/$USER/my-own-path-to-my-lsaie-scripts/assignment3a.py
    "

    # Submits the CMD to all the processes on all the nodes.
    srun bash -c "$CMD"

    echo "[sbatch-master] task finished"
    ```
    For instructional reasons, this example is verbose and doesn't do much more than print environment variables. However, it is important to understand how information flows from the node which executes the sbatch script, to the srun commands executed for each process, to the actual script that is being executed. 

    Different to the previous sbatch script we now ask for four nodes using the `--nodes` argument and we also specify that each node runs four processes through the `--ntasks-per-node` argument. The other arguments we have covered before. Notice that in an sbatch script, we have access to **job script placeholders** like `%u` or `%j` which SLURM automatically replaces/expands to the username and job id. If you are curious you can find more on the [official SLURM documentation on sbatch](https://slurm.schedmd.com/sbatch.html).

    Notice that we have to use `\` to prevent variables from being evaluated on the sbatch node when they should be evaluated on the compute node. To illustrate this we added a "wrong_hostname" command which, as we will see shortly, will be evaluated on the sbatch node and not on the compute node. Can you tell why it is not necessary to escape $USER?

2. Create a new python file `assignment3a.py` with the following content. Carefully read the comments.
    ```
    #!/usr/bin/env python3
    import os
    import socket
    import torch
    import torch.distributed as dist

    # Read environment variables that we set in the sbatch script
    master_addr = os.environ.get("MASTER_ADDR", "N/A")
    master_port = os.environ.get("MASTER_PORT", "N/A")
    world_size = int(os.environ.get("WORLD_SIZE", "N/A"))
    foobar = os.environ.get("FOOBAR", "N/A")

    # Read environment variables set by SLURM
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])

    # Read the information form the node
    hostname = socket.gethostname()

    # Each process prints a final message to confirm it didn't get stuck
    print(f"[Python] rank={rank} | host={hostname} | {master_addr}:{master_port} | {world_size} | {foobar} ")
    ```

3. Execute the sbatch script.
    ```
    sbatch run_assignment_3a.sbatch
    ```
    As you know from assignment 1, you can see your position in the queue and run-state with `squeue`. This run should take about 80 seconds to complete. Once completed, take a look at the files it created, in particular the `.out` file. Answer the following questions by looking at the slurm outputs. Which node executed the sbatch script? How many times was your python script executed?

4. In the previous example, the Python script did not communicate across ranks. We will use `torch.distributed` which abstracts away low-level details of inter-process communication and is commonly used in Pytorch-based machine learning projects. To achieve this we will expand the previous script by initialising a process group and perform a simple All-Reduce operation which will sum up a tensor across all ranks/processes.

    1. Expand your python script `assignment3a.py` with the following lines.
    ```
    # Initializes the default (global) process group
    dist.init_process_group(
        backend="nccl",  # the NCCL backend requires a GPU on each process
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
    )

    # Limit GPU allocation of this process to only one GPU
    torch.cuda.set_device(local_rank)

    # Create a float32 tensor on each rank with a single element of value 'rank' and move it to the GPU.
    local_tensor = torch.tensor([rank], dtype=torch.float32).cuda()
    print(f"[Python] rank={rank} | local_rank={local_rank} | host={hostname} | local_tensor={local_tensor.item()}")

    # Perform a sum operation across all ranks.
    dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM)
    print(f"[Python] rank={rank} | local_rank={local_rank} | host={hostname} | local_tensor_after_all_reduce={local_tensor.item()}")

    # Cleanup
    dist.destroy_process_group()
    ```

    2. Submit your sbatch script again. By looking at your logs, can you tell which rank will have the sum across all ranks?

5. Before we move on to other collective communications, we will take a look at `torchrun` which is a PyTorch-native utility for launching distributed training jobs. It provides an easier multi-node setup and integrates well with SLURM. 

    1. With torchrun, we don't have to worry about the processes that are created on a node for each GPU and instead just launch one process running torchrun for each node. For this reason, we need to instruct SLURM to only create 1 task per node. Let's change the SBATCH arguments at the top of your sbatch script to reflect that.
    ```
    #SBATCH --ntasks-per-node=1
    ```
    Now instead of using `python` to execute your python script we use `torchrun` as follows.
    ```
    # The command that will run on each node
    CMD="
    # print current environment variables
    echo \"[srun] rank=\$SLURM_PROCID host=\$(hostname) noderank=\$SLURM_NODEID localrank=\$SLURM_LOCALID\"

    # run the script
    torchrun \
        --nnodes="${SLURM_NNODES}" \
        --node_rank=\$SLURM_NODEID \
        --nproc_per_node=4 \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        /iopsstor/scratch/cscs/$USER/large-scale-ai-eng-course/assignment3c.py
    "

    # Submits the CMD to all the processes on all the nodes.
    srun bash -c "$CMD"
    ```
    Notice that each node needs to be told which rank it is so they can communicate that to the other nodes. This is why the argument `--node_rank` has `$SLURM_NODEID` escaped such that they get the rank of the compute node. `torchrun` will wait (and eventually time out) until all ranks have been connected. If we did not escape `$SLURM_NODEID` all compute nodes would be rank 0 and wait for other ranks to connect.

    2. The python script simplifies a bit since now the communication is taken care of by `torchrun`. Here it is in its entirety with the minimal necessary.
    ```
    #!/usr/bin/env python3
    import os
    import socket
    import torch
    import torch.distributed as dist

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

    # Perform a sum operation across all ranks.
    dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM)
    print(f"[Python] rank={rank} | local_tensor_after_all_reduce={local_tensor.item()}")

    # Cleanup
    dist.destroy_process_group()
    ```

    3. Submit the sbatch script as before. If your changes were successful you should see roughly the same output with an all-reduce operation over 16 ranks. 


## [2/7] All-Reduce and Measuring Throughput

You have already performed an all-reduce in the previous section. All-reduce is a common collective communication operation that combines data from all participating processes (ranks) and distributes the combined result back to every process. In deep learning, it is e.g. widely used to synchronise gradients in data-parallel training across multiple GPUs and nodes. Make the following changes to your script from the previous section.

![All-Reduce visualisation from Nvidia](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/allreduce.png)

1. Create the data directly on the GPU.
    ```
    N = 2 ** 30  # ~1.1 billion elements
    tensor = torch.full((N,), fill_value=rank, dtype=torch.float32, device="cuda")
    ```

2. To measure the time it takes we need to call `torch.cuda.synchronize`. This is because CUDA operations are asynchronous by default. When a cuda command is issued it is **queued** on the GPU and not immediately executed. `torch.cuda.synchronize` forces the program to wait until all cuda commands have completed so we know the queue is empty and our next cuda command will be executed immediately.
    ```
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
    ```

3. Add a check to see if `tensor` now has the expected value.
    ```
    expected_val = world_size*(world_size-1)/2
    assert torch.allclose(
        tensor,
        torch.full_like(tensor, expected_val)
    ), f"[Python] rank={rank} | all-Reduce mismatch: expected {expected_val}, got {tensor[0].item()} in first element."
    ```

4. Compute throughput and print it to your log file. What throughput do you get?
    ```
    total_bytes = tensor.nelement() * 4  # convert elements to bytes
    total_gbs = total_bytes / (1024**3)  # convert to GB
    throughput = total_gbs / elapsed_seconds  # GB/s

    print(f"[Python] rank={rank} | transferred {total_gbs:.2}GB | throughput={throughput:.4}GB/s")
    ```

5. When running such a benchmark, the first few runs can include hidden costs like establishing network connections and memory allocation. We simply do a few warmup iterations **before** the benchmark to get these one-time costs out of the way, so we can measure the true performance of our operations. What throughput do you get now?
    ```
    # Warmup
    for _ in range(5):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    ```

6. Note that we can also do all-reduce as an async_op and spend the time that we wait for all-reduce to complete on another operation. Try e.g. run the following to visualise the "work" which can be done during the data transfer.
    ```
    async_op = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
    while not async_op.is_completed():
        print(f"{rank}|", end='', flush=True)  # Print the rank number without a newline to simulate CPU work
        time.sleep(0.1)  # Wait for 0.1 seconds
    ```

## [3/7] Reduce and Broadcast

In contrast to all-reduce, reduce collects gradients from all ranks, computes their average, and stores the result in a single rank. Broadcast does the inverse, as it distributes its own data to all processes. This could be done to perform an update to some parameter tensor on a single master rank which first collects the gradients from all ranks and then distributes the new parameters after the update. Let's go through a toy version of this.

Reduce:

![Reduce visualisation from Nvidia](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/reduce.png)

Broadcast:

![Broadcast visualisation from Nvidia](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/broadcast.png)

1. Prepare a new sbatch and python script for a multi-node job over 4 nodes similar to the previous ones. 

2. Initialise parameters and gradients.
    ```
    N = 2 ** 30  # ~1.1 billion elements
    parameters = torch.ones((N,), dtype=torch.float32, device="cuda")
    gradients = torch.full((N,), fill_value=rank, dtype=torch.float32, device="cuda")
    print(f"[Python] rank={rank} | Initial parameters[0]={parameters[0].item()}")
    ```

3. Let's define our root-rank which will perform the parameter update and the learning it should use.
    ```
    LEARNING_RATE = 0.1 
    ROOT_RANK = 0  # Central rank for parameter updates
    ```

4. As before, synchronise and measure the time. Then perform the following reduction to the root node.
    ```
    # Perform a reduce mean operation and store the result on root_rank,
    dist.reduce(gradients, dst=ROOT_RANK, op=dist.ReduceOp.SUM)
    ```
    followed by a parameter update using the average gradient and the learning rate
    ```
    # On root rank, compute the average gradient and update parameters
    if rank == ROOT_RANK:
        gradients /= world_size  # Average the gradients
        parameters -= LEARNING_RATE * gradients  # SGD update
    ```
    and a broadcast to distribute the parameters back to all ranks.
    ```
    # Broadcast updated parameters to all ranks
    dist.broadcast(parameters, src=ROOT_RANK)
    ```

4. Use these lines to verify that parameters are the same across all ranks:
    ```
    expected_param = 1.0 - LEARNING_RATE * (world_size - 1) / 2
    assert torch.allclose(
        parameters[0],
        torch.tensor(expected_param, device="cuda")
    ), f"[Python] rank={rank} | Parameter mismatch: expected {expected_param}, got {parameters[0].item()}"
    ```

5. Calculate and print the throughput similar as before. How does the throughput compare with the previous sections?


## [4/7] Send and Receive in a Ring

Let's implement a ring communication pattern using point-to-point `send` and `recv` operations in PyTorch's distributed framework. Each rank should send data to the next rank, receive data from the previous rank, and verify the integrity of the received data. 

1. Prepare a new sbatch and python script for a multi-node job over 4 nodes similar to the previous ones. 

2. Create a tensor to send and allocate an empty tensor to receive data.
    ```
    # Create a tensor to send: filled with the sender's rank
    send_tensor = torch.full((N,), fill_value=rank, dtype=torch.float32, device="cuda")

    # Prepare a tensor to receive data
    recv_tensor = torch.zeros(N, dtype=torch.float32, device="cuda")
    ```

3. We can now use ```send``` and ```recv``` for this purpose but those operations are blocking which means that we have to perform the operations in two steps where half of the ranks are first receiving and the other half is sending.
    ```python
    if rank % 2 == 0:
        send_req = dist.send(tensor=send_tensor, dst=send_rank)
        recv_req = dist.recv(tensor=recv_tensor, src=recv_rank)
    else:
        recv_req = dist.recv(tensor=recv_tensor, src=recv_rank)
        send_req = dist.send(tensor=send_tensor, dst=send_rank)
    ```
    Make sure to perform a few warmup iterations in the same way before measuring the time.

4. In order to send and receive simultaneously (and increase our throughput) we can use ```isend``` and ```irecv```. Although those operations are async they are not upon the first call which means that we need to perform the warmup as previously.
    ```python
    send_req = dist.isend(tensor=send_tensor, dst=send_rank)
    recv_req = dist.irecv(tensor=recv_tensor, src=recv_rank)

    # Wait for both send and receive to complete
    #torch.cuda.synchronize()
    send_req.wait()
    recv_req.wait()
    print(f"[Python] rank={rank} is_complete={send_req.is_completed()}", flush=True)
    print(f"[Python] rank={rank} is_complete={recv_req.is_completed()}", flush=True)
    torch.cuda.synchronize()  # shouldn't be needed but .wait() is not behaving as expected.
    ```
    Unfortunately, the current environment contains an older PyTorch version which doesn't behave as expected. In particular, as demonstrated by the code above, the ```.wait()``` calls on the async send and receive operations is not actually blocking which is the reason why we need to ```synchronize``` before measuring the time. The PyTorch version in this environment also fails upon calling ```destroy_process_group()``` process group which is due to an issue that was recently fixed in PyTorch. 

4. Compute and print the throughput similar to before. Explain if it has higher, lower, or roughly equal throughput to the previous all-reduce experiment and why that might be.


## [5/7] AllGather and ReduceScatter 

In this section, you will implement two additional collective communication operations using PyTorch's distributed framework with the NCCL backend: AllGather and ReduceScatter. These operations are fundamental in distributed computing, particularly in scenarios involving model parallelism and data partitioning.

AllGather: Each process starts with a unique subset of data. AllGather concatenates these subsets from all processes and distributes the complete combined data back to every process. After AllGather, each process holds the entire data.

![AllGather visualisation from Nvidia](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/allgather.png)

ReduceScatter: This operation is a combination of reduction and scatter. It first reduces data across all processes (e.g., summing tensors) and then scatters the reduced result into disjoint chunks, with each process receiving a unique portion of the result.

![ReduceScatter visualisation from Nvidia](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/reducescatter.png)

1. Prepare a new sbatch and python script for a multi-node job over 4 nodes similar to the previous ones. 

2. We will first do an AllGather operation. Prepare a float32 tensor with `2**27` elements and a two-dimensional tensor to hold the data that we gather.
    ```
    # Total parameters size remains the same but splits across ranks
    N = 2 ** 27  # ~0.13 billion elements

    ## ----- AllGather 
    # Each process starts with a unique subset of data
    send_tensor= torch.full((N,), fill_value=rank, dtype=torch.float32, device="cuda")

    # Create a tensor to store the data collected from all rank
    recv_tensor = torch.zeros((world_size, N), dtype=torch.float32, device="cuda")
    ```

3. Sync all processes again and measure the time before calling the all_gather operation.
    ```
    dist.all_gather(tensor_list=[recv_tensor[i] for i in range(world_size)],
                    tensor=send_tensor)
    ```
4. To verify that you have collected the tensors from all ranks you can print the recv_tensor tensor averaged across the `N` axis.

5. Next, we will add a ReduceScatter operation which is essentially a Reduce operation followed by a Scatter operation. 
    ```
    # Each process starts with the full dataset
    send_tensor = torch.full((world_size * N,), fill_value=rank, dtype=torch.float32, device=device)

    # And we create a tensor to hold 1/world_size part of it.
    recv_tensor = torch.zeros((N,), dtype=torch.float32, device=device)
    ```

6. As before, measure the time and set a barrier. Then we can perform a reduce_scatter as follows.
    ```
    dist.reduce_scatter(tensor=recv_tensor,
                        tensor_list=[send_tensor[i*N:(i+1)*N] for i in range(world_size)],
                        op=dist.ReduceOp.SUM)
    ```

7. Let's now print on each rank the mean of the recv_tensor so we can verify in the logs that each rank has received the correct data.
    ```
    print(f"[Python ReduceScatter] rank={rank} | transferred {total_gbs:.2}GB | throughput={throughput:.4}GB/s | recv_tensor.mean()={recv_tensor.mean()}")
    ```

8. Looking at the logs of the ReduceScatter operations you should see an average of `120.0` on all ranks. If you run it a few times, what do you notice? 

## [6/7] Process Groups

In PyTorch, a **process group** is a fundamental concept in its distributed training framework.
It is a logical grouping of processes that can communicate with each other during distributed training using collective operations. 
So far we have been using the **global process group**, meaning all processes, one for each rank, in the training job can communicate with each other. However, in more complex scenarios, multiple process groups can be useful. Let's go through an instructive example. 

1. We have seen two different ways of initialising the global process (PG) group using `dist.init_process_group()`. Given this PG, we can e.g. create a PG for each node.
    ```
    nodes_groups = [
        dist.new_group([0,1,2,3]),
        dist.new_group([4,5,6,7]),
        dist.new_group([8,9,10,11]),
        dist.new_group([12,13,14,15]),
    ]
    ```

2. Now let's create, as before, a tensor filled with the rank
    ```
    # Total parameters size remains the same but splits across ranks
    N = 2 ** 30  # ~1.1 billion elements

    # Each process starts with data of its rank
    tensor = torch.full((N,), fill_value=rank, dtype=torch.float32, device="cuda")
    ```

3. To run an operation within the group, the executing process has to be part of the group that it passes using the `group` argument. Given the groups as in step 1, you can compute the group index with `rank // 4`.
    ```
    # Executes the reduce op on the group to which this process belongs.
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=node_groups[rank // 4])
    ```

4. As before, make sure to use dist.barrier() and cuda.synchronize to properly measure time. Then print for each rank the tensor average and throughput.
    ```
    print(f"[Python] rank={rank} | transferred {total_gbs:.2}GB | throughput={throughput:.4}GB/s | tensor.mean()={tensor.mean()}")
    ```

5. Now redo the experiment but do it with a less natural process group. To get the group index from the rank you can simply use `rank % 4`.
    ```
    node_groups = [
        dist.new_group([0,4,8,12]),
        dist.new_group([1,5,9,13]),
        dist.new_group([2,6,10,14]),
        dist.new_group([3,7,11,15]),
    ]
    ```

## [7/7] Handin

To pass this assignment you need to submit a short **markdown** report including the script to generate a single plot comparing the throughput (GB/s) versus data size for three communication patterns: global all-reduce across all ranks, all-reduce within the node-local process group (ranks 0-3, 4-7, etc.), and all-reduce within cross-node process groups (ranks 0,4,8,12, etc.). The x-axis should use a logarithmic scale showing data sizes from 2^10 to 2^32 elements. The y-axis should show throughput in GB/s. Include at least 8 measurements per communication pattern, evenly spaced on the log scale. Briefly explain any significant throughput differences you observe between the three patterns and explain why they occur. Make sure you do warmup runs before running your benchmark.



