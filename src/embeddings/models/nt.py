'''Code to load and use models from the Nucleotide-Transformer family.'''

import torch
import numpy
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

def get_model_name(version: str):
    return f'InstaDeepAI/nucleotide-transformer-{version}'

def get_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
    )
    return tokenizer

def get_model_random(model_name: str):
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    model = AutoModelForMaskedLM.from_config(
        config, 
        trust_remote_code=True,
    )
    return model

def get_model_pretrained(model_name: str):
    model = AutoModelForMaskedLM.from_pretrained(
        model_name, 
        trust_remote_code=True,
    )
    return model

def tokenize(tokenizer, sequences: list[str]):
    tokens_ids = tokenizer.batch_encode_plus(
        sequences, 
        return_tensors="pt", 
        padding=True,
    )["input_ids"]

    return tokens_ids

def get_embeddings(model, tokenizer, sequences: list[str]):
    '''
    Return all embeddings (a list containing pooled embeddings for each layer).

    Output is [nlayers * batch, dmodel].
    '''
    input = tokenize(tokenizer, sequences)
    attention_mask = input != tokenizer.pad_token_id

    output = model(
        input,
        attention_mask=attention_mask,
        encoder_attention_mask=attention_mask,
        output_hidden_states=True,
    )['hidden_states']

    batch = len(sequences)
    nlayers = len(output)
    embeddings = []
    
    for layer_embeddings in output:
        mask = torch.unsqueeze(attention_mask, dim=-1).float()
        pooled_embeddings = torch.sum(mask * layer_embeddings, dim=1) / torch.sum(mask, dim=1)
        embeddings.append(pooled_embeddings.detach().cpu().numpy())

    embeddings = numpy.array(embeddings)
    #embeddings = numpy.transpose(embeddings, (1, 0, 2))
    #embeddings = numpy.reshape(embeddings, (nlayers*batch, -1))
    return embeddings
