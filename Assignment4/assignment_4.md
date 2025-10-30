# Assignment 4: Data and Tensor Parallelism

In this assignment, we will build on what we have learned and explore the first parallelism strategies in the context of gradient-based optimisation. You will implement simple but instructive examples of batch accumulation, data parallel, and tensor parallel.

## [1/6] Setup and Utility Functions

1. As before, let's begin with preparing an sbatch script which uses `torchrun` to execute the data parallel python script that we will now develop across four nodes. There is nothing new here so given the previous assignments you should be able to create one. To keep things simple in this assignment, we will always run on 4 nodes and implement all compute strategies in a single file.

2. For this assignment, the distributed environment init and a few additional utility functions are used for several examples so it makes sense to move them into a static file called `utils.py` which we will simply import for each of the implementations. 
    ```bash
    mkdir -p assignment_4
    cd assignment_4
    touch utils.py
    ```

3. We need to initialise our distributed environment. Since we are launching our script with `torchrun` we have access to environment variables such as `RANK`, `LOCAL_RANK`, and `WORLD_SIZE`. Let's write a function that initialises the process group and assigns the proper GPU to each process. We'll keep things a bit verbose on purpose so the logs are more interpretable. 
    ```python
    import os
    import torch
    import torch.distributed as dist

    def init_distributed():
        """
        Initialise the distributed environment.
        Assumes that environment variables RANK, LOCAL_RANK, and WORLD_SIZE are set.
        """
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        node_id = os.environ.get("SLURM_NODEID", "N/A")
        
        # Set the current device for this process
        torch.cuda.set_device(local_rank)

        # Initialise the process group with NCCL backend (requires Nvidia GPUs)
        dist.init_process_group(backend="nccl")
        
        print(f"[Distributed Init] Rank {rank} initialized on {node_id} on GPU {local_rank}.")
        dist.barrier()
        if rank == 0:
            print(f"[Rank {rank}] All ranks ready!")
        return rank, local_rank, world_size
    ```

4. In this assignment we'll be working with dummy data. The following function will give us a batch of dummy inputs and targets needed to "train" our model.
    ```python
    def create_batch(batch_size, input_dim, output_dim, seed=42, device="cuda"):
        """
        Create synthetic input and target tensors for a batch.
        
        Parameters:
            batch_size (int): Number of examples in the batch
            input_dim (int): Dimension of each input example
            seed (int): Random seed for reproducibility
            device (str): Device to create the tensors on ("cuda" or "cpu")
        
        Returns:
            tuple: (inputs, targets)
                - inputs: Tensor of shape (batch_size, input_dim) containing random values
                - targets: Tensor of shape (batch_size,) for regression tasks
        """
        torch.manual_seed(seed)
        # Create input tensor
        inputs = torch.randn(batch_size, input_dim, device=device)
        
        # Create regression targets
        targets = torch.randn(batch_size, output_dim, device=device)
        
        return inputs, targets
    ```

5. Let's also add a helper function to check the shape of tensor during runtime. This is quite helpful as comments of shape sizes can be inaccurate. 
    ```python
    def check(tensor, expected_shape):
        """
        Checks that the tensor's shape matches the expected shape.
        
        Parameters:
            tensor (torch.Tensor): The tensor to check.
            expected_shape (list or tuple): The expected shape. For any dimension,
                                            a value <= 0 is treated as a wildcard.
                                            
        Raises:
            AssertionError: If tensor is not a torch.Tensor or if the actual shape
                            does not match the expected shape.
                            
        Example:
            >>> x = torch.randn(32, 64, 10)
            >>> check(x, [32, 64, 10])  # Passes - exact match
            >>> check(x, [32, -1, 10])  # Passes - middle dimension treated as wildcard
            >>> check(x, [32, 128, 10])  # Raises AssertionError - middle dimension mismatch
        """
        if tensor is None:
            return

        # Ensure the provided tensor is a torch.Tensor.
        assert isinstance(tensor, torch.Tensor), "SHAPE GUARD: Provided tensor is not a torch.Tensor!"

        # Detach the tensor to avoid issues with in-place modifications (useful for torch.compile).
        actual_shape = list(tensor.detach().shape)

        # Ensure expected_shape is a list or tuple.
        assert isinstance(expected_shape, (list, tuple)), "SHAPE GUARD: expected_shape must be a list or tuple!"
        assert len(expected_shape) == len(actual_shape), (
            f"SHAPE GUARD: Expected shape length {len(expected_shape)} but got {len(actual_shape)} with shape {actual_shape}."
        )

        for idx, (actual_dim, expected_dim) in enumerate(zip(actual_shape, expected_shape)):
            # Skip the check for dimensions with a non-positive expected size (wildcard).
            if expected_dim <= 0:
                continue
            assert actual_dim == expected_dim, (
                f"SHAPE GUARD: At dimension {idx}: expected {expected_dim}, but got {actual_dim} (actual shape: {actual_shape})."
            )
    ```

6. In the next few sections of the assignment, we'll run the exact same computation in different ways. We can use the following method to compare the resulting tensors to see if our computations are equivalent in each case.
    ```python
    def compare_tensors(tensor1, tensor2, tol=1e-5, prefix=""):
        """
        Simple comparison of two tensors, printing basic difference statistics.
        
        Parameters:
            tensor1 (torch.Tensor): First tensor to compare
            tensor2 (torch.Tensor): Second tensor to compare
            tol (float): Tolerance for considering values as close
        """
        # Calculate differences
        abs_diff = (tensor1 - tensor2).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        
        # Simple check if tensors are close
        is_close = torch.allclose(tensor1, tensor2, rtol=tol, atol=tol)
        
        # Print brief comparison summary
        rank = dist.get_rank() if dist.is_initialized() else 0
        prefix = f"[{prefix}]" if prefix else ""
        print(f"{prefix}[Rank {rank}] Tensors match: {is_close} | Max diff: {max_diff:.6e} | Mean diff: {mean_diff:.6e}", flush=True)
    ```

## [2/6] The Single GPU Baseline
Let's create an ```assignment4.py``` file that will contain all our code. Each section will be a standalone function that will perform a single gradient step on a linear map using random data as input and target. The goal is that all strategies compute the same updated weight matrix. To this end, we'll begin with a single GPU baseline. 

1. We'll write all following code into our ```assignment4.py``` file. We begin with the relevant imports from PyTorch and our ```utils.py``` module followed by our distributed initialisation.
    ```python
    from utils import init_distributed, create_batch, check, compare_tensors

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.distributed as dist

    rank, local_rank, world_size = init_distributed()
    ```

2. For educational reasons the model used will be small linear layer. We'll use the following custom implementation to simplify the initialisation of the model across ranks.
    ```python
    # Define global parameters
    global_batch_size = 128   # Must be divisible by world_size (e.g., if world_size=16, each gets 8)
    local_batch_size = global_batch_size // world_size
    input_dim = 64
    output_dim = 32
    seed = 42

    class CustomLinearLayer(nn.Module):
        """
        A linear layer.

        weight matrix W has shape [in_dim, out_dim]
        activation matrix X has shape [bsz, in_dim]

        out = X @ W which as shape [bsz, out_dim]
        """
        def __init__(self, weight: torch.Tensor):
            super(CustomLinearLayer, self).__init__()
            self.W = nn.Parameter(weight)
            self.in_dim = weight.shape[0]
            self.out_dim = weight.shape[1]

        def forward(self, X):
            local_bsz = X.shape[0]
            check(X, (local_bsz, self.in_dim))

            # Batched matrix-vector multiplication
            # this could be replaced with matmul or @
            X = torch.einsum("bi,ij->bj", X, self.W)

            check(X, (local_bsz, self.out_dim))
            return X
    ```

3. Next we will add to our script the ```single_step``` function which will compute a single gradient-based update. Complete the lines marked with TODOs.
    ```python
    ### Part 1: We compute the reference weight on a single GPU.
    def single_step(seed=42, device="cuda") -> torch.Tensor:
        """
        Educational example of performing a single gradient step.
        """
        # Set the seed for reproducibility
        torch.manual_seed(seed)
        
        # Generate a weight matrix
        initial_weight = torch.randn(input_dim, output_dim)
        
        # Create the custom linear model using the provided weight matrix.
        model = CustomLinearLayer(initial_weight).to(device)
        
        # Set up the SGD optimizer with learning rate 0.5
        optimizer = optim.SGD(model.parameters(), lr=0.5)

        # Create the loss function
        loss_fn = nn.MSELoss(reduction="mean")
        
        # Create a synthetic batch of data with global_batch_size elements
        inputs, targets = # TODO (1 line)
        check(inputs, (global_batch_size, input_dim))
        check(targets, (global_batch_size, output_dim))
        
        # Perform a forward pass through the model we defined above.
        outputs = # TODO (1 line)
        check(outputs, (global_batch_size, output_dim))
        
        # Compute the MSE loss using loss_fn defined above by taking the average over the target and batch dimension.
        loss = # TODO (1 line)
        check(loss, [])

        # Reset gardients of all parameters to 0
        # TODO (1 line)

        # compute gradients
        # TODO (1 line)

        # perform a parameter update
        # TODO (1 line)
        
        # Return the updated weight matrix (detached from the computation graph).
        return initial_weight, model.W.detach()
    ```

4. With your implementation of the previous function we can now perform a single parameter step on a single rank and compute the expected result. We will use ```updated_weight``` as our ground truth in the next sections.
    ```python
    if rank == 0:
        print(f"[Rank {rank}] Compute the updated matrix which should be different from the initial weight matrix.")
        initial_weight, updated_weight = single_step()
        compare_tensors(initial_weight, updated_weight.cpu())
    else:
        # On all other ranks we create a tensor placeholder so we can distribute the updated_weight to all ranks
        updated_weight = torch.zeros(input_dim, output_dim, device="cuda")

    # distribute updated weight to all ranks to enable a comparison with the baseline later on
    dist.broadcast(updated_weight, src=0)
    ```

5. Remember to end your script as follow to ensure the processes are all terminated.
    ```python
    # Cleanup
    dist.destroy_process_group()
    print(f"[Rank {rank}] done")
    ```

## [3/6] Batch Accumulation
Before we begin to use parallel compute, we'll perform parallelism through time, i.e. compute the gradient estimate sequentially. 

1. Let's create a new function in ```assignment4.py``` for this purpose. The function is very similar to before so it's recommended to create a copy of the previous function and make the respective edits.
    ```python
    def single_step_with_grad_accumulation(seed=42, device="cuda", accumulation_steps=4) -> torch.Tensor:
        """
        Educational example of performing a single gradient step with gradient accumulation.
        """
    ```

2. In particular, we'll still load the synthetic data in the same way but we can now use the ```microsteps``` argument to compute the ```micro_batch_size```.
    ```python
        # Create a synthetic batch of data
        inputs, targets = create_batch(global_batch_size, input_dim, output_dim, seed=seed, device=device)
        check(inputs, (global_batch_size, input_dim))
        check(targets, (global_batch_size, output_dim))
        
        # Calculate the micro batch size
        micro_batch_size = global_batch_size // accumulation_steps
    ```

3. In contrast to the previous section, after zeroing the gradients of the optimiser, we'll now perform a for loop over the data and perform backward calls on each micro batch in sequence. Each ```.backward()``` will add the resulting gradient to the ```.grad``` attribute of a parameter tensor.
    ```python
    # Reset gradients before accumulation starts
    optimizer.zero_grad()

    # Perform gradient accumulation over multiple smaller batches
    for i in range(accumulation_steps):
        # Calculate the start and end indices for this micro-batch
        start_idx = # TODO
        end_idx = # TODO
        
        # Slice the original inputs and targets to get this micro-batch
        micro_inputs = inputs[start_idx:end_idx]
        micro_targets = targets[start_idx:end_idx]
        check(micro_inputs, (micro_batch_size, input_dim))
        check(micro_targets, (micro_batch_size, output_dim))

        # Perform a forward pass through the model
        micro_outputs = # TODO
        check(micro_outputs, (micro_batch_size, output_dim))
        
        # Compute the loss for this micro-batch
        micro_loss = # TODO
        check(micro_loss, [])

        # Scale the loss to maintain the same gradient magnitude regardless of accumulation steps. It is numerically advantagous to divide by the number of steps before computing the sum.
        scaled_loss = # TODO

        # Compute gradients (backward pass)
        # The gradients are accumulated (summed) in param.grad
        scaled_loss.backward()

    # After accumulating gradients from all micro-batches, update parameters
    optimizer.step()

    # Return updated weight matrix
    return model.W.detach()
    ```

4. Let's now add the following lines to our script to compare on rank 0 if the new model parameters using ```single_step_with_grad_accumulation``` matches the previous results. If your code is correct, this should be the case. 
    ```python
    if rank == 0:
        print(f"[Rank {rank}] Compute the updated weight using batch accumulation. They should match.")
        batch_accum_weight = single_step_with_grad_accumulation()
        compare_tensors(updated_weight.cpu(), batch_accum_weight.cpu())
    ```

## [4/6] Data Parallel

In data parallelism, each rank only processes a subset of the entire data. At the beginning of the script we defined ```global_batch_size = 128```. Hence, each rank will process a slice of size `global_batch_size // world_size`.

1. As before, we'll add a new function to perform a single gradient step. We are not going to use batch accumulation anymore so we can begin using a copy of our previous ```single_step``` function.
    ```python
    ### Part 3: We compute the updated weight using data parallelism
    def data_parallel_single_step(seed=42, device="cuda") -> torch.Tensor:
        """
        Educational example of performing a single gradient step using data parallelism.
        Each process handles a subset of the global batch.
        """
    ```

2. Recall that the script will be executed by each rank so each rank will randomly initialise it's own weight matrix. To ensure they all end up with the same initial weight matrix, a requirement to also compute the same updated weight matrix, the script defines a seed using ```torch.manual_seed```. Alternatively we could have created the matrix on one rank and distributed it using a broadcast operation. 
    ```python
    # Set the seed for reproducibility
    # We need to ensure all processes start with the same weight
    torch.manual_seed(seed)

    # Generate a weight matrix
    initial_weight = torch.randn(input_dim, output_dim)

    # Alternatively we could broadcast the tensor from rank 0 to all other processes
    # initial_weight = initial_weight.to(device)
    # dist.broadcast(initial_weight, src=0)
    ```

3. Similarly as before we will load the global data on each rank and then extract from it the rank-specific local data before perform a local forward and backward pass.
    ```python
    # Create a synthetic batch of data with the same seed across all workers
    # Then each process will handle a subset of the data based on rank
    full_inputs, full_targets = create_batch(global_batch_size, input_dim, output_dim,
        seed=seed, device=device)
    
    # Calculate start and end indices for this process's portion of data
    start_idx = rank * local_batch_size
    end_idx = start_idx + local_batch_size
    
    # Get local batch by slicing the full batch based on rank
    local_inputs = full_inputs[start_idx:end_idx]
    local_targets = full_targets[start_idx:end_idx]
    check(local_inputs, (local_batch_size, input_dim))
    check(local_targets, (local_batch_size, output_dim))
    
    # Reset gradients before forward/backward pass
    optimizer.zero_grad()
    
    # Perform a forward pass through the model with the local batch
    local_outputs = model(local_inputs)
    check(local_outputs, (local_batch_size, output_dim))
    
    # Compute the MSE loss for the local batch
    local_loss = loss_fn(local_outputs, local_targets)
    check(local_loss, [])

    # Compute gradients (backward pass)
    local_loss.backward()
    ```

4. Next, before we update the model parameters we need to average them across all ranks. Which communication primitive do you need to use here? 
    ```python 
    # Synchronize gradients across all processes
    for param in model.parameters():
        # Sum the gradients across all processes
        # TODO (1 line)
        # Average the gradients by dividing by world_size
        param.grad.div_(world_size)  # Good to know: in pytorch func_ are in-place operations. 
    
    # Perform parameter update - all processes will have the same update now
    optimizer.step()
    
    # Return the updated weight matrix
    return model.W.detach()
    ```

5. Adding the following lines to our script allows us to run our gradient update in a data parallel fashion and verify the output.
    ```python
    if rank == 0:
        print(f"[Rank {rank}] Compute the updated weight using data parallelism.")
    data_parallel_weight = data_parallel_single_step()

    # Compare on all ranks
    compare_tensors(updated_weight.cpu(), data_parallel_weight.cpu(), prefix="DataParallel")
    ```

## [5/6] Tensor Parallel: A Full Column Parallel Example 
In the previous section we implemented an education example of data parallelism, where each rank processes a different subset of the input data. 
This approach scales well but requires replicating the full model on every device which limits the size of our models and forces us to scale up the global batch size.
In this section, we'll look at tensor parallel which is a method to split a module of a model across multiple ranks.

Here we will now implement the "full" column parallel example using a final AllGather as illustarted in the slides. 

1. In order to shard a linear map over its columns, we'll need to perform a broadcast and AllGather operation. These operations are not differentiable so we'll have to implement them ourselves by inheriting from ```torch.autograd.Function``` before we use use them for our tensor parallel step function.
    ```python
    class BroadcastParallel(torch.autograd.Function):
        """
        Identity in forward pass, AllReduce in backward pass.
        Identical to function f in the Megatron paper (https://arxiv.org/abs/1909.08053).
        """
        @staticmethod
        def forward(ctx, x):
            # The forward pass is an identity because we assume the data is already sharded.
            return x

        @staticmethod
        def backward(ctx, grad_output):
            if world_size == 1:
                return grad_output
            dist.all_reduce(grad_output, op=ReduceOp.SUM)
            return grad_output

    class GatherParallel(torch.autograd.Function):
        """Gather in forward pass, split in backward pass."""
        @staticmethod
        def forward(ctx, x):
            if world_size == 1:
                return x

            x = x.contiguous()
            x_list = [torch.empty_like(x) for _ in range(world_size)]
            x_list[rank] = x        
            dist.all_gather(x_list, x)
            out = torch.cat(x_list, dim=-1).contiguous()
            return out

        @staticmethod
        def backward(ctx, grad_output):
            if world_size == 1:
                return grad_output
            local_dim = grad_output.shape[-1] // world_size
            grad_output_split = torch.split(grad_output, local_dim, dim=-1)
            return grad_output_split[rank].contiguous()
    ```

2. Next, we'll need to write a column-parallel version of our previous model ```CustomLinearLayer```. Make sure to fill out the TODO lines appropriately.
    ```python
    class FullColumnParallelLinear(nn.Module):
        """
        A column-parallel linear layer with a final gather.
        
        weight matrix W has shape [in_dim, out_dim]
        activation matrix X has shape [bsz, in_dim]

        out = X @ W which as shape [bsz, out_dim]
        
        In a column-parallel linear layer, the weight matrix is now split along the 
        output (column) dimension such that each rank holds a shard of shape 
        [in_dim, out_dim / world_size]. The results are all [bsz, out_dim] 
        and need to be summed across all ranks.
        
        Parameters:
            weight (torch.Tensor): Full weight matrix of shape (out_dim, in_dim).
            world_size (int): Total number of processes.
            rank (int): The current process rank.
        """
        def __init__(self, weight: torch.Tensor, world_size: int, rank: int):
            super(FullColumnParallelLinear, self).__init__()
            in_dim, out_dim = weight.shape
            assert out_dim % world_size == 0, "out_dim must be divisible by world_size"
            
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.world_size = world_size
            self.rank = rank
            self.local_out_dim = out_dim // world_size
            
            # Partition the weight along columns for this rank.
            start = rank * self.local_out_dim
            end = start + self.local_out_dim
            self.W = nn.Parameter(weight[:, start:end].clone().contiguous())

        def forward(self, X):
            """
            Forward pass for the column-parallel linear layer.
            
            Parameters:
                X (torch.Tensor): Input tensor of shape (batch_size, in_dim).
                
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, out_dim).
            """
            local_bsz = X.shape[0]
            check(X, (local_bsz, self.in_dim))
            
            # distribute X in the forward pass and collect the errors from all ranks in the backward pass
            X = # TODO
            
            # Batched matrix multiplication
            local_out = torch.einsum("bi,ij->bj", X, self.W).contiguous()
            check(local_out, (local_bsz, self.local_out_dim))
            
            # Collect the outputs of the linear map in the forward pass and keep only the rank specific errors in the backward pass. 
            out = # TODO
            check(out, (local_bsz, self.out_dim))
            
            return out
    ```

3. Now we can finally implement our full column-parallel linear step. For simplicity, we'll perform all steps similarly as before.
    ```python 
    def full_column_parallel_single_step(seed=42, device="cuda"):
        """
        Performs one gradient update using tensor parallelism with column-wise splitting.
        """
        torch.manual_seed(seed)
        initial_weight = torch.randn(input_dim, output_dim)

        # Instantiate the column-parallel linear model.
        model = FullColumnParallelLinear(initial_weight, world_size, rank).to(device)
        
        # Set up optimiser as previously
        optimizer = optim.SGD(model.parameters(), lr=0.5)
        loss_fn = nn.MSELoss(reduction="mean")

        # Get a batch
        full_inputs, full_targets = create_batch(global_batch_size, input_dim, output_dim, seed=seed, device=device)
        
        # Forward pass: Each process slices its corresponding input columns.
        outputs = model(full_inputs)
        check(outputs, (global_batch_size, output_dim))
        
        # Compute the loss using the full output.
        loss = loss_fn(outputs, full_targets)
        
        # Backward pass.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ```

4. After performing an optimiser update, each rank holds an updated shard of the model. We need to collect them to return the global matrix for the comparison. Fill out the TODO line appropriately.
    ```python
    # Each rank now holds its shard of the updated weight of shape (input_dim, local_output_dim).
    local_updated_weight = model.W.detach()
    check(local_updated_weight, (input_dim, output_dim // world_size))

    # Gather weight shards from all processes along the column dimension.
    # To do so we first need to create a list of zero tensors on the GPU so we can call all_gather
    weight_shards = # TODO (1 line)
    dist.all_gather(weight_shards, local_updated_weight)

    # Concatenate the shards along columns to reconstruct the full updated weight.
    global_updated_weight = torch.cat(weight_shards, dim=1)
    check(global_updated_weight, (input_dim, output_dim))

    return global_updated_weight
    ```

5. We can again check our implementation by comparing the updated weight matrices. If your implementation is correct ```compare_tensors``` should return ```Tensors match: True``` for each rank. 
    ```python
    if rank == 0:
        print(f"[Rank {rank}] Compute the updated weight using tensor parallelism (FullColumnParallelLinear).")

    column_parallel_weight = full_column_parallel_single_step()
    compare_tensors(updated_weight.cpu(), column_parallel_weight.cpu(), prefix="FullColumnParallel")
    ```

## [6/6] Handin
Following the sections above you should have created the files ```utils.py``` and a longer ```assignment4.py``` which implements and runs all 4 parts (single gpu baseline, batch accumulation, data parallel, and tensor parallel). Your handin should consist of a zip archive of those files **including** the stderr (```.err```) and stdout (```.log```) files of a run. If you followed the instructions above, your stdout file should contain the prints that confirm the correctness of your code for the latter 3 parts. You do not need to submit a report or any other files.