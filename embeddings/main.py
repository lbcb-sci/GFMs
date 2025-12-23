import torch
from datasets.genomic_benchmarks import get_dataset, DATASETS_OF_INTEREST
from models.nt import get_model_name, get_model, get_tokenizer, get_embeddings

VERSION = 'v2-50m-multi-species'

model_name = get_model_name(VERSION) 
model = get_model(model_name)
tokenizer = get_tokenizer(model_name)

sequences = ["ATTCCGATTCCGATTCCG", "ATTTCTCTCTCTCTCTGAGATCGATCGATCGAT"]

embeddings = get_embeddings(model, tokenizer, sequences)

print(f"Embeddings[0] shape: {embeddings[0].shape}")
print(f"Embeddings[-1] shape: {embeddings[-1].shape}")
print(f"len(Embeddings): {len(embeddings)}")
#print(f"Embeddings per token: {embeddings}")

## Add embed dimension axis
#attention_mask = torch.unsqueeze(attention_mask, dim=-1)

## Compute mean embeddings per sequence
#mean_sequence_embeddings = torch.sum(attention_mask*embeddings, axis=-2)/torch.sum(attention_mask, axis=1)
#print(f"Mean sequence embeddings: {mean_sequence_embeddings}")
