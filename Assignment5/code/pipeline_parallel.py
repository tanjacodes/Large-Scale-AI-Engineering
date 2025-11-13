import copy
import torch

from typing import List

from torch import nn
from torch import distributed as dist
# Q7: Complete the conditions of the `if` statements within each `operation`
def pipeline_communicate(operation, pp_process_group, tensor=None, shapes=None):
    """
    P2P communication helper for pipeline parallelism.

    operation: one of 'recv_forward', 'send_forward', 'recv_backward', 'send_backward'
    pp_process_group: pipeline process group
    tensor: for send operations the tensor to send; for recv operations ignored (we allocate)
    shapes: shape to allocate for recv operations if tensor is not provided
    """

    # NOTE: `src` & `dest` MUST be global ranks, hence we do "src = dist.get_global_rank..."
    pp_rank = dist.get_rank(pp_process_group)
    pp_prev_rank = pp_rank - 1
    pp_next_rank = pp_rank + 1

    pp_is_first_stage = pp_rank == 0
    pp_is_last_stage = pp_rank == dist.get_world_size(pp_process_group) - 1

    if operation == 'recv_forward':
        # First stage has no forward recv
        if pp_is_first_stage:
            return None
        # allocate a tensor that will carry activations; preserve autograd link by requires_grad=True
        tensor = torch.empty(shapes, requires_grad=True, device="cuda")
        src = dist.get_global_rank(pp_process_group, pp_prev_rank)

    elif operation == 'send_forward':
        # Last stage has no forward send
        if pp_is_last_stage:
            return
        # tensor must be provided by caller for send
        dest = dist.get_global_rank(pp_process_group, pp_next_rank)

    elif operation == 'recv_backward':
        # Last stage has no backward recv (it produces loss and grads)
        if pp_is_last_stage:
            return None
        tensor = torch.empty(shapes, requires_grad=True, device="cuda")
        src = dist.get_global_rank(pp_process_group, pp_next_rank)

    elif operation == 'send_backward':
        # First stage has no backward send (no previous stage to send to)
        if pp_is_first_stage:
            return
        dest = dist.get_global_rank(pp_process_group, pp_prev_rank)

    else:
        raise ValueError(f"Unknown pipeline operation: {operation}")

    # decide peer rank and send/recv op
    print_shapes = shapes if shapes else tensor.shape
    is_send = operation.startswith('send')
    peer_rank = dest if is_send else src

    # Non-blocking P2P wrapper via batch_isend_irecv
    op = dist.P2POp(dist.isend if is_send else dist.irecv, tensor, peer_rank)
    [req.wait() for req in dist.batch_isend_irecv([op])]
    torch.cuda.synchronize()
    return tensor if not is_send else None


# Q5: Develop this function
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
class PipelineStage(nn.Module):
	"""
	Implements pipeline parallelism by distributing model layers across multiple GPUs.
	Each GPU processes a subset of the model's layers in a pipeline fashion.
	"""
	def __init__(self, model, number_of_layers, pp_rank, pp_world_size):
		super().__init__()
		# Determine which layers should be assigned to this GPU
		self.layer_distribution = distribute_layers(number_of_layers, pp_rank, pp_world_size)
		# Assign relevant decoder layers to this GPU
		self.pp_stage_layers = nn.ModuleList([copy.deepcopy(model.layers[i]) for i in self.layer_distribution])
    
	def forward(self, x):
		for layer in self.pp_stage_layers:
			x = layer(x)
		return x
    
	def backward(self, input_tensor, output_tensor, output_tensor_grad):
		"""
		Backward pass for this pipeline stage.
		Computes gradients for assigned layers using received gradient from next stage.
		"""
		if input_tensor is not None: input_tensor.retain_grad()
		if output_tensor_grad is None:
			#match gradients of mean()
			output_tensor_grad = torch.ones_like(output_tensor, memory_format=torch.preserve_format)
			output_tensor_grad = output_tensor_grad / output_tensor.numel()
		torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad, retain_graph=False, create_graph=False)
		return input_tensor.grad if input_tensor is not None else None 
