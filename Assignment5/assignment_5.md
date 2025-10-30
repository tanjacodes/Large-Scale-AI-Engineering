# Assignment 5: Pipeline Parallelism in Depth

In this exercise, we will delve into pipeline parallelism, a sharding technique that involves dividing a model's layers across multiple devices. We will explore the constraints of this technique, the communication patterns, and how the backward pass is computed. Finally, we will verify that both the outputs and the gradients from each layer match between the pipelined and non-pipelined models.

## Introduction

The source code for this assignment can be found in `/capstor/store/cscs/ethz/large-sc-2/assignment_5`. Throughout the exercise, you will need to complete the missing lines of code explained in each task. There are a total of 10 exercises, which must be completed in order. At the end of the exercise, you should be able to verify that you can shard a model across multiple devices and that this does not affect the model's quality.

We will start by checking the **Init** section in `test.py`. Read the arguments of the `main` function carefully, as they will accompany us throughout this assignment. As observed, in this section, we will prepare the environment to execute this PyTorch distributed script. Remember that it will be necessary to run the script using `torchrun`, following the rule of thumb of one process per GPU.

**Q1:** To keep things simple, we have established that we will not support other sharding dimensions such as data parallelism or tensor parallelism. In all pipeline parallel ranks, we will create the full model and then shard it. Because of this, all processes will receive the same seed and will therefore generate the same weights and inputs. However, outside this fictitious scenario created for the exercise, answer the following questions and explain why:

1. In each process, should the seed across the pipeline parallel dimension be the same or different?
2. In each process, should the seed across the data parallel dimension be the same or different?
3. In each process, should the seed across the tensor parallel dimension be the same or different? Hint: Remember that when applying tensor parallelism we shard most of the linear layers but we don't shard the layernorm layers.
4. Now imagine using all three sharding techniques simultaneously. Have you detected any inconsistencies in your answers? Can you think of a way to solve this?

**Q2:** When applying pipeline parallelism to a model, we need to ensure that the workload of each stage is as balanced as possible, since the performance of the pipeline will always be constrained by the slowest stage. Another commonly used technique in this type of scenario is to divide the work into smaller units (micro-batches) to process more data concurrently across the different pipeline stages. Assert that the arguments satisfy these two conditions.

## Model

**Q3:** In this exercise, we will use the model defined in `model.py`. Examine the model definition. What shape will the tensors that we communicate across the pipeline stages have?

## Baseline Forward & Backward

To ensure that our pipeline parallelism implementation is correct, we will execute a forward and backward pass of a batch of data with a non-sharded copy of the model in all processes and compare it with the sharded version.

**Q4:** Write the code that performs the following:
1. Fetch a batch of data from the dataloader.
2. Move the batch of data to the appropriate device.
3. Compute the forward pass.
4. Compute the backward pass. Note that since we do not have a label tensor, we will compute a fake loss value using the `.mean()` of the output tensor.

## Pipelined Forward & Backward

In this assignment, we will implement the **All forward, All backward** schedule (Figure 1). This is the simplest schedule, as its name suggests: first, compute all the micro-batches forward passes, then compute all the backward passes.

<img src="https://i.ibb.co/d4xB7D8Y/Pipeline-schedule-1-625x185.png">

> Figure 1: All Forward, All Backward pipeline schedule


To do this, we will first develop the `PipelineStage` class, which is essentially a wrapper around our model defined in the previous section that contains only the necessary layers in each pipeline stage. The forward pass is straightforward, but note that we have also had to define the backward pass. This is because, when sharding the model in the pipeline dimension, the `torch.autograd` engine can no longer automatically trace the computation graph, so we must explicitly call it in each pipeline stage.

**Q5:** Carefully examine the `pipeline_parallel.py` file and implement the `distribute_layers` function, which will return a list of the layers contained in each pipeline stage.

One characteristic of sharding a model with pipeline parallelism is that not all stages require a training dataloader, as they receive inputs from a previous stage in the forward pass and gradients from the subsequent stage during the backward pass.

**Q6:** Which ranks require the training dataloader?

As mentioned, when leveraging pipeline parallelism, we need to handle communication of both intermediate activations and gradients. To do this, we have developed the `pipeline_communicate` function.

**Q7:** Observe the `pipeline_communicate` function in `pipeline_parallel.py`. Using Figure 1 for reference, explain when each of the four possible `operation` cases will occur. You can sketch in the image which communications each GPU will carry out and when. Keep in mind that those operations are P2P involving 2 pipeline stages. Correctly complete the four `if` statements within each `operation`. This function returns `None` when not performing any P2P communication.

**Q8:** Write the code that performs the following:
1. Fetch a batch of data from the dataloader if needed.
2. Move the batch from the dataloader OR the activations from the previous pipeline parallel stage to the appropriate GPU.
3. Compute the forward pass.
4. Compute the backward pass. Use the `.mean()` of the output tensor as the loss function.
Note that the first layer might need special treatment since its input tensor doesn't have gradients enabled.

## Pipelined vs Non-Pipelined Model

Finally, we will use the `torch.testing.assert_close` function to verify that both the outputs and the gradients from the weights of each layer match between the pipelined and non-pipelined models.

**Q9:** Check the model outputs in the required rank. Do they pass the `assert_close` test?

**Q10:** Check the gradients of the required layers. Remember that in each pipeline stage, you only have a subset of the original model’s layers, so you only need to verify the gradients of the weights for the layers contained in each rank.

## Deliverable

Submit a single markdown file (.md) with the answers to Q1 and Q7, along with code snippets for the remaining questions. Do *not* submit multiple files.

