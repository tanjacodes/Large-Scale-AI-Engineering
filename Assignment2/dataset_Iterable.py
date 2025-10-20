import pyarrow.parquet as pq 
from torch.utils.data import Dataset 
from transformers import AutoTokenizer 


class IterableParquetDataset(IterableDataset):
    def __init__(
        self,
        parquet_file: str,
        tokenizer,
        sequence_length: int,
        bos_token_id: int = 1):
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
	while len(self.token_buffer) < self.sequence_length + 1:
	    if self.current_index < self.real_length:
	        next_sample = str(self.parquet_ds["text"][self.current_index])
		self.token_buffer.extend(self.tokenizer.endcode_plus(next_sample)["input_ids"])
		self.token_buffer.append(self.bos_token_id)
		self.current_index += 1
	    else: 
		break
	    
	    #In case we are out of tokens
	    if len(self.token_buffer) < self.sequence_length + 1 
		raise StopIteration
	
	    tokens_to_release = self.token_buffer[:self.sequence_length + 1]
	    self.token_buffer = self.token_buffer[self.chunk_len :]
	    inputs = torch.LongTensor(tokens_to_release[:-1])
	    #labels are shifted by 1
	    labels = torch.LongTensor(tokens_to_release[1:])
	    return inputs, labels
	    
