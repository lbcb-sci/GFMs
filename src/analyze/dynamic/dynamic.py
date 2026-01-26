import numpy
import torch
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, PreTrainedTokenizer
from ckatorch import CKA

import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import get_plots_path
from .data import CudaWrapper, get_dataset_dna, get_dataset_text

@torch.no_grad()
def dynamic_analysis(models_dict: dict, tokenizers_dict: dict, logger):

    for (run, models), (_, tokenizers) in zip(models_dict.items(), tokenizers_dict.items()):
        assert run == _

        bert1: BertForMaskedLM = models[0]
        bert2: BertForMaskedLM = models[1]
        tokenizer: PreTrainedTokenizer = tokenizers[0]

        preprocess = lambda batch: tokenizer(
            batch['text'], 
            truncation=True, 
            padding='max_length', 
            max_length=512,
            return_tensors='pt',
        )

        if 'dna' in run:
            logger.info(f' collecting dataset dna...')
            dataset = get_dataset_dna()
            remove = ['text']
        
        elif 'text' in run:
            logger.info(f' collecting dataset text...')
            dataset = get_dataset_text()
            remove = ['text', 'url', 'id', 'title']

        encoded = dataset.map(preprocess, batched=True, remove_columns=remove)
        encoded.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        dataloader = DataLoader(CudaWrapper(encoded), batch_size=128, num_workers=0, shuffle=True)
        # num workers MUST be 0 here

        nlayers = bert1.config.num_hidden_layers
        layers = [f"bert.encoder.layer.{i}.output.dense" for i in range(nlayers)]

        # minibatch CKA
        cka = CKA(bert1, bert2, layers=layers, first_name='bert1', second_name='bert2', device='cuda')
        cka_matrix = cka(dataloader, epochs=10)

        logger.info(run)
        logger.info(cka_matrix)

        cka_matrix = cka_matrix.detach().cpu().numpy()
        numpy.savetxt(run, cka_matrix)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cka_matrix,
            annot=True, fmt='.2f', cmap='coolwarm',
            vmin=0.0, vmax=1.0, square=True, linewidths=0.5,
            cbar_kws={'label': 'CKA Similarity'},
            xticklabels=range(1, nlayers+1), yticklabels=range(1, nlayers+1)[::-1]
        )
        plt.title(run, fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(get_plots_path() / (run + '.png'))
        plt.close()
