# Assignment 2: PyTorch Fundamentals and Language Modelling

In this assignment, you'll review the main PyTorch components needed to train a language model. You'll explore PyTorch datasets, dataloaders, model architecture, training loops, and performance analysis. The goal is to understand how these components work together and make a few small improvements.

## [1/7] Working with a Tokenizer

Let's start by exploring how tokenizers and datasets work together in PyTorch. Tokenizers convert text to token IDs that models can process, while datasets provide a structured way to access your training data.

1. Connect to clariden and create a new project folder in your iopsstor home directory.
   ```
   ssh clariden
   mkdir -p /iopsstor/scratch/cscs/$USER/assignment-2
   cd /iopsstor/scratch/cscs/$USER/assignment-2
   ```
	Remember, in assignment 1 you might have made a symlink called `scratch` in your home directory pointing to `/iopsstor/scratch/cscs/$USER/`. In that case you can replace those instances by `~/scratch`.
2. Let's activate our conda environment and install the following packages.
   ```
   conda activate
   pip install torch transformers pyarrow datasets
   ```

3. Create a new file `tokensiation_example.py` and add the following lines to load a tokeniser using the `AutoTokenizer` class from huggingface's y
   ```
   from transformers import AutoTokenizer

   # Initialize the tokenizer
   tokenizer = AutoTokenizer.from_pretrained("unsloth/Mistral-Nemo-Base-2407-bnb-4bit")
   ```

4. We can use `encode` and `decode` to convert a string to token ids and back. You can execute this script on the login node since it barely needs resources. You can also run them on your machine, but that might need some additional setup steps depending on your local environment.
   ```
   # Example of converting text to tokens and back
   text = "Hello, I am a language model."

   # Convert text to tokens (IDs)
   tokens = tokenizer.encode(text)
   print(f"Tokens: {tokens}")

   # Convert tokens back to text
   decoded_text = tokenizer.decode(tokens)
   print(f"Decoded text: {decoded_text}")
   ```

## [2/7] Create a Dataset

1. Create a file named `dataset.py` and add this dataset definition.
    ```
    import pyarrow.parquet as pq 
    from torch.utils.data import Dataset 
    from transformers import AutoTokenizer 

    class ParquetDataset(Dataset):
      def __init__(self, parquet_file: str, tokenizer: str, sequence_length: int, training_samples: int):
        self.parquet_ds = pq.read_table(parquet_file, memory_map=True)
        self.real_length = len(self.parquet_ds)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.training_samples = training_samples

      def __len__(self):
        return self.training_samples
      
      def __getitem__(self, idx: int):
        sample_str = str(self.parquet_ds["text"][idx % self.real_length])
        return self.tokenizer.encode_plus(sample_str,
                                          max_length=self.sequence_length + 1,
                                          padding='max_length',
                                          truncation=True,
                                          padding_side="right")
    ```

2. Create a script that loads a tokeniser as shown before. Use the following sequence length and dataset path if you run it on the login node. 

3. Create a new dataset object using `ParquetDataset` using the provided path. If you are running your scripts locally, you'll have to download the training data (2.3GB) and change the paths accordingly.
    ```
    # Create dataset instance
    dataset_path = "/capstor/store/cscs/ethz/large-sc-2/datasets/train_data.parquet"
    sequence_length = 4096

    # Create dataset (only requesting 1 sample)
    dataset = ParquetDataset(
        parquet_file=dataset_path,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        training_samples=1
    )

    # Get the first sample
    sample = dataset[0]
    ```

3. **DELIVERABLE 1:** Provide the string of the first 200 decoded tokens of the first sample. 

## [3/7] Create a Collator
Next, we'll add a data collator which processes multiple samples into batches that can be fed to our model. 

1. Add the following code to your `dataset.py` file.
    ```
    from dataclasses import dataclass
    from typing import List, Dict
    import torch

    @dataclass
    class CollatorForCLM:
      sequence_length: int
      pad_token_id: int
      def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.LongTensor([examples[i]["input_ids"] for i in range(len(examples))])  # (b, s+1)

        inputs = input_ids[:, :-1].clone()
        labels = input_ids[:, 1:]

        # For padding tokens, mask the loss
        labels[labels == self.pad_token_id] = -100

        assert inputs.shape[1] == labels.shape[1] == self.sequence_length
        assert inputs.shape == labels.shape

        return inputs, labels
    ```

2. Expand the script you created to sample the dataset and expand it with the following collator definition to create a dataloader object using `torch.utils.data.DataLoader` and define `batch_size = 32` and make sure to update the `ParquetDataset` instance to provide 32 samples instead of just 1 .
    ```
    # Create collator
    collator = CollatorForCLM(sequence_length=sequence_length, pad_token_id=tokenizer.pad_token_id)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

    # Get a batch using a for loop 
    for batch_inputs, batch_labels in dataloader:
        # Print shapes
        print(f"Input shape: {batch_inputs.shape}")
        print(f"Labels shape: {batch_labels.shape}")

        # Count ignored tokens in the loss calculation
        ignored_count = (batch_labels == -100).sum().item()
        total_label_tokens = batch_labels.numel()
        print(f"Ignored tokens in loss: {ignored_count} out of {total_label_tokens} ({ignored_count/total_label_tokens*100:.2f}%)")

        # Only process the first batch
        break
    ``` 

3. **DELIVERABLE 2:**  Given the defined collator and the script above. What is the total number of tokens in a batch, and how many of those are currently ignored due to padding? Why is this a problem?

## [4/7] Create a More Efficient Iterable Dataset
Currently, padding is a significant source of inefficiency. Each padded token consumes memory and compute resources but doesn't contribute to model training. This is especially problematic when working with varying-length documents that need to be padded to a fixed sequence length.

To solve this problem, you have to create an `IterableParquetDataset` that concatenates text sequences together and eliminates padding. Instead of padding each individual document, we'll join multiple documents together, only adding special tokens (the Begin of Document token, ID 1) between them, creating a continuous stream of useful tokens that we can easily collate.

1. Create a copy of `ParquetDataset` in `dataset.py` and name it `IterableParquetDataset` which inherits from `torch.utils.data.IterableDataset`. 

    ``` 
    class IterableParquetDataset(IterableDataset):
        def __init__(
            self,
            parquet_file: str,
            tokenizer,
            sequence_length: int,
            bos_token_id: int = 1
        ):
            self.parquet_ds = pq.read_table(parquet_file, memory_map=True)
            self.real_length = len(self.parquet_ds)
            self.tokenizer = tokenizer
            self.sequence_length = sequence_length
            self.bos_token_id = bos_token_id
            self.current_index = 0
            self.token_buffer = []
        
        def __iter__(self):
            # Reset buffer and index when starting a new iteration
            self.token_buffer = []
            self.current_index = 0
            return self
        
        def __next__(self):
            # Keep filling a buffer until we have enough tokens for a new sample.
            # Mask the loss for each token following the BoS token using -100 index.
            
            # Add your implementation here

            yield inputs, labels
    ``` 

2. Implement your solution using a buffer. Keep adding to the buffer until it is large enough to serve a sample. To mask the target of the BoS token recall that the target of the BoS token is the next token.

3. **DELIVERABLE 3:** Provide only your IterableParquetDataset class implementation in an otherwise empty python file.

## [5/7] Explore the Model
Now let's explore the model architecture we'll be using for training. The model is a decoder-only transformer similar to those used in modern LLMs. 

1. Make a copy of the model implementation. 
    ```
    cp /capstor/store/cscs/ethz/large-sc-2/assignment_2/model.py /iopsstor/scratch/cscs/$USER/assignment-2/.
    ```

2. Create a script to load the model using the following model config. The model is fairly large, do not run this on your personal machine.  
    ```
    from model import Transformer, TransformerModelArgs

    model_config = TransformerModelArgs(
            dim=4096,
            n_layers=32,
            n_heads=32,
            n_kv_heads=8,
            ffn_dim_multiplier=1.3,
            multiple_of=1024,
            rope_theta=500000,
            vocab_size=tokenizer.vocab_size,
            seq_len=4096,
    )

    model = Transformer(model_config)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    ```

2. **DELIVERABLE 4:**: What is the total parameter count (no rounding) of the model defined above? Run this on a computenode as it requires significant memory to instantiate such a large model.

## [6/7] Train a Model

Now let's examine the training loop and learning rate scheduler used for training our model.

1. Make a copy of the trainer implementation, an additional utils file, and an sbatch script to launch a training run. 
    ```
    cp /capstor/store/cscs/ethz/large-sc-2/assignment_2/train.py /iopsstor/scratch/cscs/$USER/assignment-2/.
    cp /capstor/store/cscs/ethz/large-sc-2/assignment_2/utils.py /iopsstor/scratch/cscs/$USER/assignment-2/.
    cp /capstor/store/cscs/ethz/large-sc-2/assignment_2/submit-llama3.sh /iopsstor/scratch/cscs/$USER/assignment-2/.
    ```

2. We will use a custom environment for this task. Make a copy of the environment file to your scratch partition. In the sbatch script, adjust the environment path to point to this location. In the script you need to use your actual username for this case, as `%u` or `$USER` doesn't work.
```
cp /capstor/store/cscs/ethz/large-sc-2/environment/ngc_pt_jan.toml /iopsstor/scratch/cscs/$USER/.
```

3. Within `train.py`, you find a simple implementation of a training loop. Go through all three files and make sure you understand them. 

4. We have included logging for token throughput, hardware throughput, and the achieved Model FLOPS Utilization (MFU) in the training script. Hardware throughput refers to the number of Floating Point Operations a GPU executes. It is typically measured in TFLOPs per second, where one TFLOP equals 10^12 floating point operations.  

MFU measures how efficiently a GPU executes the theoretical peak FLOPs when training or running inference on a model. It is defined as the "Achieved TFLOPs" divided by "Theoretical Peak FLOPs".


MFU helps assess how well the hardware is utilized. A high MFU (e.g., 40-50%) indicates good efficiency, while a low MFU (e.g., <20%) suggests bottlenecks, such as memory bandwidth limitations, inefficient kernels, or communication overhead.

4. **DELIVERABLE 5:** Benchmark the effects of `--fused-optimizer`, `--compile`, and doubling the  `--sequence-length` to 4096. Report the different throughputs you achieve. Give an explanation as to why each of these configurations improves or decreases performance compared to the baseline.

## [7/7] Performance Profiling

  Lastly, we'll explore how to use the NVIDIA NSYS Profiler. You will need to download NVIDIA Nsight Systems in your *local machine* to visualize the traces [https://developer.nvidia.com/nsight-systems/get-started]. You don't have to download *anything* on the cluster.

  1. We've already implemented the profiler integration for you in the `train.py` file. To extract a trace, we need to specify the `--profile` flag in the `train.py` script.  To launch the program using the NSYS profiler with sbatch, set `TRAINING_CMD` as follows in your `submit-llama3.sh`:
      ```bash
      nsys profile -s none -w true \
        --trace='nvtx,cudnn,cublas,cuda' \
        --output=/iopsstor/scratch/cscs/$USER/assignment-2/nsys-trace.nsys-rep \
        --force-overwrite true \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop -x true numactl --membind=0-3 python3 $ASSIGNMENT_DIR/train.py --profile
      ```
		

  2. **DELIVERABLE 6:**  Extract a trace of the program and visualize it on your computer. Attach a screenshot showing both CPU activities and GPU activities for the three GEMMs present in a single layer of the FeedForward block. Compare the actual execution time with the theoretical performance you would achieve using NVIDIA's advertised specifications (Peak BF16 Performance: 989 TFLOP/s, Peak Memory Bandwidth: 4 TB/s).
