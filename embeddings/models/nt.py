'''Code to load and use models from the Nucleotide-Transformer family.'''

from transformers import AutoModelForMaskedLM, AutoTokenizer

def get_model_name(version: str):
    return f'InstaDeepAI/nucleotide-transformer-{version}'

def get_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
    )
    return tokenizer

def get_model(model_name: str):
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
    '''Return all embeddings (a list containing embeddings at each layer).'''

    input = tokenize(tokenizer, sequences)

    attention_mask = input != tokenizer.pad_token_id

    torch_outs = model(
        input,
        attention_mask=attention_mask,
        encoder_attention_mask=attention_mask,
        output_hidden_states=True,
    )

    embeddings = [layer_embs.detach().numpy() for layer_embs in torch_outs['hidden_states']]
    return embeddings
