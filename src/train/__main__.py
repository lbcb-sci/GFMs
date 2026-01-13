import torch, numpy, random
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    TrainingArguments, BertForMaskedLM,
    BertConfig, Trainer, set_seed,
)

from src.train.tokenizer import train_bpe_tokenizer

def main():
    dataset = load_dataset("salesforce/wikitext", "wikitext-2-v1")
    dataset_train = dataset["train"]
    dataset_eval  = dataset["validation"]

    tokenizer = train_bpe_tokenizer(make_iterator(dataset_train), 1000)

    def preprocess(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    train_encoded = dataset_train.map(preprocess, batched=True, remove_columns=["text"])
    eval_encoded  = dataset_eval.map(preprocess, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)

    batch_size = 1024 
    matrices = []
    N = 5
    epochs = 5

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

        print(config.initializer_range)

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
            dataloader_num_workers=8,
            num_train_epochs=epochs,
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

        embeddings = model.bert.embeddings.word_embeddings.weight.detach() 
        embeddings = F.normalize(embeddings, p=2, dim=1)

        distances = 1.0 - (embeddings @ embeddings.T)
        diag = torch.diag(distances)
        assert diag.allclose(torch.zeros_like(diag, dtype=torch.float32), rtol=1e-5, atol=1e-5)
        matrices.append(distances)

    for i, a in enumerate(matrices):
        for b in matrices[i+1:]:
            diff = a - b
            numel = diff.numel()

            rmse = torch.linalg.norm(diff) / numel**0.5
            mse = (diff.pow(2).sum() / numel)
            mae = diff.abs().mean()

            print(rmse, mse, mae)

    tensor = torch.stack(matrices)
    print(tensor.shape)

    std = tensor.std(dim=0)
    print(std.shape)
    print(std.mean())

    print(tensor[:, 0, 1])
    print(tensor[:, 0, 2])
    print(tensor[:, 0, 3])
    print(tensor[:, 10, 100])
    print(tensor[:, 100, 65])
    print(tensor[:, 78, 89])

def make_iterator(dataset):
    def text_iterator():
        for example in dataset:
            txt = example['text']
            if isinstance(txt, str) and txt.strip(): yield txt
    return text_iterator

if __name__ == "__main__":
    main()
