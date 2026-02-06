'''
Showing that text BERT models learn meaningful relationships even
in their very first static word embeddings layer.
'''

import torch
import argparse
from transformers import BertForMaskedLM, AutoTokenizer
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

@torch.autograd.inference_mode()
def find_closest(token: str, bert: BertForMaskedLM, tokenizer, n: int = 5):
    assert token in tokenizer.get_vocab()

    out = tokenizer(token, return_tensors='pt')['input_ids']
    embedding1 = bert.bert.embeddings.word_embeddings(out)[0]

    similarities = []

    for t in tokenizer.get_vocab():

        if t == token or t in token or token in t:
            continue

        out = tokenizer(t, return_tensors='pt')['input_ids']
        embedding2 = bert.bert.embeddings.word_embeddings(out)[0]

        sim = pairwise_cosine_similarity(embedding1, embedding2)[0][0].item()
        similarities.append((t, sim))

    return list(map(lambda p: p[0], (sorted(similarities, key=lambda p: p[1], reverse=True))))[:n]

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
        print(f'Closest tokens from [{token}]: {closest}')

if __name__ == '__main__':
    main()
