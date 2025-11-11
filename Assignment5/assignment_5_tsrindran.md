# Assignment 5: Pipeline Parallelism in Depth

## Question 1

1. No, it should not be the same because each pipeline stage holds a different subset of layers of the model. If they used the same seed, they would all initialize identical weights, but each stage needs unique weights for its own layers. 
2. Yes, it should be the same otherwise the calculations are not coherent. Each data parallel replica holds a complete copy of the model and processes different batches of data. The gradients need to remain synchronized when averaged, therefore we need identical model initialization and all data-parallel ranks must start with the same weights.
3. For linear layers, we need different seeds to initialize the shards of ranks that hold different shards of the same parameter tensor.
For the LayerNorm, biases or embeddings that reamin replicated, all ranks must share the same seed so these parameters are inizialized identically.
4. For some techniques we need all processes to have the same seed, for other techniqzes we need all processes to have different seeds. 
In order to solve this we could define seeds at different levels such that we can reproduce the seed whenever needed and such that we can use a different seed when needed.   


## Question 3
The size of the tensors will be:

(micro_batch_size, sequence_length, hidden_size)




## Question 6

Only the first pipeline rank (rank 0) requires the training dataloader, because it’s the only one that directly consumes input data. All other ranks receive intermediate activations from the previous stage. 
