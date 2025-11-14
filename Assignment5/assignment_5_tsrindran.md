# Assignment 5: Pipeline Parallelism in Depth

## Question 1

1. No, it should not be the same because each pipeline stage holds a different subset of layers of the model. If they used the same seed, they would all initialize identical weights, but each stage needs unique weights for its own layers. 
2. Yes, it should be the same otherwise the calculations are not coherent. Each data parallel replica holds a complete copy of the model and processes different batches of data. The gradients need to remain synchronized when averaged, therefore we need identical model initialization and all data-parallel ranks must start with the same weights.
3. For linear layers, we need different seeds to initialize the shards of ranks that hold different shards of the same parameter tensor.
For the LayerNorm, biases or embeddings that reamin replicated, all ranks must share the same seed so these parameters are inizialized identically.
4. For some techniques we need all processes to have the same seed, for other techniqzes we need all processes to have different seeds. 
In order to solve this we could define seeds at different levels such that we can reproduce the seed whenever needed and such that we can use a different seed when needed. 

## Question 2
```python
assert number_of_layers % pp == 0 
assert global_batch_size // micro_batch_size >= pp and global_batch_size % micro_batch_size == 0 
```

## Question 3
```python
tensor_shapes = (micro_batch_size, sequence_length, hidden_size)
```

## Question 4
```python
batch = next(train_dl_iterator)
# 2. Move it to the GPU
batch = batch.cuda(non_blocking = True)
# 3. Compute the forward pass
output = model(batch)


output_tensors_no_pp.append(output.detach().clone()) # NOTE(tj.solergibert) To check PP vs NON-PP outputs!
# 4. Compute the backward pass
output.mean().backward() 
```


## Question 5
```python
def distribute_layers(num_layers: int, pp_rank: int, pp_world_size: int) -> List[int]:
    """
    Distribute model layers across GPUs as evenly as possible.
    Returns a list with the layer indices that should be processed by this GPU.
    """
    # Compute how many layers per stage (base) and the remainder
    layers_per_stage = num_layers // pp_world_size
    remainder = num_layers % pp_world_size

    # First `remainder` stages get one extra layer
    if pp_rank < remainder:
        start_idx = pp_rank * (layers_per_stage + 1)
        end_idx = start_idx + (layers_per_stage + 1)
    else:
        start_idx = remainder * (layers_per_stage + 1) + (pp_rank - remainder) * layers_per_stage
        end_idx = start_idx + layers_per_stage

    layers_in_current_stage = list(range(start_idx, end_idx))
    return layers_in_current_stage

```
## Question 6
```python
if device_mesh["pp"].get_local_rank() == 0: 
    train_dl_iterator = iter(input)
else:
    train_dl_iterator = None
```
## Question 7
recv_forward: In the forward phase, each stage (except the first) receives input activations from the stage before it. The first stage does not perform this receive because it reads data directly from the dataloader.

send_forward: Still in the forward phase, each stage (except the last) sends its output activations to the next stage. The final stage does not send anything forward because it produces the model’s output.

recv_backward: In the backward phase, each stage (except the last) receives gradient tensors from the stage ahead of it. The last stage does not receive gradients, since it computes the loss and initiates backpropagation.

send_backward: Also in the backward phase, each stage (except the first) sends gradient tensors back to the previous stage. The first stage does not send gradients further back because there is no earlier stage.
## Question 8
```python
if device_mesh["pp"].get_local_rank() == 0: 
            input_tensor = next(train_dl_iterator)
input_tensor = input_tensor.cuda(non_blocking=True)
output = model(input_tensor)
pipeline_communicate(operation='send_forward', pp_process_group=device_mesh["pp"].get_group(), tensor=output)
        
output_tensors_pp.append(output.detach().clone()) # NOTE(tj.solergibert) To check PP vs NON-PP outputs!
        
# Compute loss on the last stage
# 4. Compute the loss in the required stage
if device_mesh["pp"].get_local_rank() == dist.get_world_size(device_mesh["pp"].get_group()) -1: 
    output.mean().backward()
```

## Question 9
```python
if device_mesh["pp"].get_local_rank() == dist.get_world_size(device_mesh["pp"].get_group()) -1:
```

## Question 10
```python
for pp_stage_layer in model_stage.pp_stage_layers:
    if pp_stage_layer.fc1.weight.grad is not None:
        torch.testing.assert_close( pp_stage_layer.fc1.weight.grad,model.layers[pp_stage_layer.layer_idx].fc1.weight.grad) 
    if pp_stage_layer.fc2.weight.grad is not None:
        torch.testing.assert_close( pp_stage_layer.fc2.weight.grad,model.layers[pp_stage_layer.layer_idx].fc2.weight.grad) 
```

