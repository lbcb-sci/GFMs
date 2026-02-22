from transformers import BertForMaskedLM, AutoTokenizer
from huggingface_hub import login

login()

indices = [1, 3, 6, 9, 11]

pairs = [
    ('text', 'bpe'),
    ('dna', 'bpe'),
    ('dna', 'kmer'),
]

for data, tok in pairs:

    for i, idx in enumerate(indices):

        path = f'../runs/90M_{data}_{tok}/{idx}'

        model = BertForMaskedLM.from_pretrained(
            path, local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True
        )

        # TODO/ push both models to huggingface hub
        repo_id = f'mrochk/bert-90M-{data}-{tok}-{i+1}'

        tokenizer.push_to_hub(repo_id)
        model.push_to_hub(repo_id)

        loaded = BertForMaskedLM.from_pretrained(repo_id, local_files_only=False, force_download=True)
        loaded = AutoTokenizer.from_pretrained(repo_id, local_files_only=False, force_download=True)

        print(f'model {data} {tok} {i}, {idx} ok.')
