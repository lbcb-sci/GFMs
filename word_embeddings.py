'''
Showing that text BERT models learn meaningful relationships even
in their very first static word embeddings layer.
'''

import torch
import argparse
from transformers import BertForMaskedLM, AutoTokenizer
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

@torch.autograd.inference_mode()
def find_closest(token: str, bert: BertForMaskedLM, tokenizer):
    assert token in tokenizer.get_vocab()

    out = tokenizer(token, return_tensors='pt')['input_ids']
    embedding1 = bert.bert.embeddings.word_embeddings(out)[0]

    closest = []
    max_sim = float('-inf')

    for t in tokenizer.get_vocab():
        out = tokenizer(t, return_tensors='pt')['input_ids']
        embedding2 = bert.bert.embeddings.word_embeddings(out)[0]

        sim = pairwise_cosine_similarity(embedding1, embedding2)[0][0].item()

        if sim > max_sim and t != token: 
            max_sim = sim
            closest.append(t)
        
    return list(reversed(closest[1:]))

def main():
    parser = argparse.ArgumentParser(description='Showing some static word embeddings examples.')
    parser.add_argument('--path', type=str)
    path = parser.parse_args().path

    bert = BertForMaskedLM.from_pretrained(path, local_files_only=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

    tokens = [
        'fish',
        'America',
        'football',
        'France',
        'computer',
        'students',
    ]

    for token in tokens:

        closest = find_closest(token, bert, tokenizer)
        print(f'Closest tokens from [{token}]: {closest[:4]}')

if __name__ == '__main__':
    main()
