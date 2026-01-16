import torch, numpy, random
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    TrainingArguments, BertForMaskedLM,
    BertConfig, Trainer, set_seed,
    EarlyStoppingCallback,
)

from src.train.tokenizer import train_bpe_tokenizer
from src.datasets import nt_tasks

def make_iterator_llm(dataset):
    def text_iterator():
        for example in dataset:
            txt = example['text']
            if isinstance(txt, str) and txt.strip(): yield txt
    return text_iterator

def make_iterator_glm(dataset):
    def text_iterator():
        for example in dataset:
            txt = example[0]
            if isinstance(txt, str) and txt.strip(): yield txt
    return text_iterator

def get_cosine_distances(model):
    model.eval()
    with torch.no_grad():
        emb = model.bert.embeddings.word_embeddings.weight.detach()  # [V, D]

        # optional: z-score across tokens
        mean = emb.mean(dim=0, keepdim=True)
        std = emb.std(dim=0, unbiased=False, keepdim=True)
        emb = (emb - mean) / (std + 1e-8)

        # L2-normalize for cosine geometry
        emb = F.normalize(emb, p=2, dim=1)

        sim = emb @ emb.T
        dist = 1.0 - sim
        diag = torch.diag(dist)
        assert diag.allclose(torch.zeros_like(diag), rtol=1e-5, atol=1e-5)
        return dist

def matrix_distance_correlation(A, B, eps=1e-8):
    # A, B: [V, V], symmetric, same tokens, same order
    assert A.shape == B.shape
    V = A.size(0)
    iu = torch.triu_indices(V, V, offset=1)  # upper triangle, no diag
    a = A[iu[0], iu[1]].view(-1)
    b = B[iu[0], iu[1]].view(-1)

    a = a - a.mean()
    b = b - b.mean()
    num = (a * b).sum()
    den = (a.pow(2).sum().sqrt() * b.pow(2).sum().sqrt()).clamp_min(eps)
    return num / den        # Pearson r in [-1, 1]

#def analyze_embeddings(models):
    #matrices = [get_cosine_distances(model) for model in models]

    #for i, a in enumerate(matrices):
        #for b in matrices[i+1:]:
            #diff = a - b
            #numel = diff.numel()
            #rmse = torch.linalg.norm(diff) / numel**0.5
            #mse = (diff.pow(2).sum() / numel)
            #mae = diff.abs().mean()
            #print(f'metrics {i}')
            #print(rmse, mse, mae)

    #tensor = torch.stack(matrices)
    #std = tensor.std(dim=0)
    #print('std:')
    #print(std.mean())

    #print('examples:')
    #print(tensor[:, 0, 1])
    #print(tensor[:, 0, 2])
    #print(tensor[:, 0, 3])
    #print(tensor[:, 10, 100])
    #print(tensor[:, 100, 65])
    #print(tensor[:, 78, 89])

def analyze_embeddings(models):
    matrices = [get_cosine_distances(model) for model in models]

    print("pairwise correlations:")
    for i, A in enumerate(matrices):
        for j, B in enumerate(matrices[i+1:], start=i+1):
            r = matrix_distance_correlation(A, B)
            print(f"{i} vs {j}: r = {r.item():.4f}")

def train_llms(N: int, vocab_size: int = 1000, epochs: int = 5):
    dataset = load_dataset("salesforce/wikitext", "wikitext-2-v1")
    dataset_train = dataset["train"]
    dataset_eval  = dataset["validation"]

    tokenizer = train_bpe_tokenizer(make_iterator_llm(dataset_train), vocab_size)

    def preprocess(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    train_encoded = dataset_train.map(preprocess, batched=True, remove_columns=["text"])
    eval_encoded  = dataset_eval.map(preprocess, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)

    batch_size = 64 
    models = []

    for seed in range(N):

        config = BertConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024,
            max_position_embeddings=512,
            initializer_range=random.normalvariate(0.02, 0.001),
        )

        # making sure model gets different initialization every time!
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        set_seed(seed)

        model = BertForMaskedLM(config)

        training_args = TrainingArguments(
            seed=seed,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=0.001,
            dataloader_num_workers=8,
            num_train_epochs=10,
            save_strategy='no',
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_encoded,
            eval_dataset=eval_encoded,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        trainer.train()
        models.append(model)

    return models

def train_glms(N: int, epochs: int = 5, vocab_size: int = 1000):
    task = 'H2AFZ'
    dataset_train = nt_tasks.get_dataset(task, split='train')
    dataset_test  = nt_tasks.get_dataset(task, split='test')

    tokenizer = train_bpe_tokenizer(make_iterator_glm(dataset_train), vocab_size)

    def preprocess(batch):
        #return tokenizer(batch[0], truncation=True, padding="max_length", max_length=128)
        return tokenizer(batch[0])

    train_encoded = [preprocess(x) for x in dataset_train]
    test_encoded = [preprocess(x) for x in dataset_test]

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15,
    )

    batch_size = 64
    models = []

    for seed in range(N):
        config = BertConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024,
            max_position_embeddings=512,
            initializer_range=random.normalvariate(0.02, 0.001),
        )

        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        set_seed(seed)

        model = BertForMaskedLM(config)

        training_args = TrainingArguments(
            seed=seed,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            dataloader_num_workers=8,

            learning_rate=0.001,

            num_train_epochs=50,

            save_strategy="epoch",
            eval_strategy="epoch",

            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_encoded,
            eval_dataset=test_encoded,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=2,
                early_stopping_threshold=0.0,
            )],
        )

        trainer.train()
        models.append(model)

    return models

def correlations_for_group(models):
    mats = [get_cosine_distances(m) for m in models]
    rs = []
    for i, A in enumerate(mats):
        for j, B in enumerate(mats[i+1:], start=i+1):
            rs.append(matrix_distance_correlation(A, B).item())
    return torch.tensor(rs)

def main():
    glms = train_glms(10)
    llms = train_llms(10)

    r_glm = correlations_for_group(glms)
    r_llm = correlations_for_group(llms)

    print(f"GLM r mean = {r_glm.mean().item():.2f} / std: {r_glm.std().item()}")
    print(f"LLM r mean = {r_llm.mean().item():.2f} / std: {r_llm.std().item()}")

if __name__ == '__main__': main()
