'''Safety check to make sure models are really different.'''

from torch.linalg import norm
from transformers import BertForMaskedLM
from src.utils import DATA_TOKENIZER_PAIRS, N

def get_huggingface_paths() -> list[str]:
    N = 5; username = 'mrochk'
    result = {'text': {'bpe': []}, 'dna': {'bpe': [], 'kmer': []}}

    for data, tok in DATA_TOKENIZER_PAIRS:
        for idx in range(1, N+1):
            path = f'{username}/bert-90M-{data}-{tok}-{idx}'
            result[data][tok].append(path)

    return result

def load_models(paths: dict, device) -> dict:
    models = {'text': {'bpe': []}, 'dna': {'bpe': [], 'kmer': []}}

    for data, tok in DATA_TOKENIZER_PAIRS:
        for path in paths[data][tok]:
            model = BertForMaskedLM.from_pretrained(path, device_map=device).eval()
            models[data][tok].append(model)

    return models

def relative_error(a, b):
    return  (2*norm(a - b)) / (norm(a) + norm(b))

def main():
    paths = get_huggingface_paths()
    models = load_models(paths, 'cuda')

    models = models['dna']['kmer']

    for i in range(N):
        for j in range(i+1, N):

            model1: BertForMaskedLM = models[i]
            model2: BertForMaskedLM = models[j]

            p1 = model1.bert.encoder.layer[0].intermediate.dense.weight[0]
            p2 = model2.bert.encoder.layer[0].intermediate.dense.weight[0]
            e = relative_error(p1, p2)
            print(e)

            p1 = model1.bert.encoder.layer[0].attention.self.query.weight[200]
            p2 = model2.bert.encoder.layer[0].attention.self.query.weight[200]
            e = relative_error(p1, p2)
            print(e)

if __name__ == '__main__': main()
