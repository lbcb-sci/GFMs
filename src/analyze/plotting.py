'''Functions for plotting, quite messy.'''

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from src.utils import get_plots_path


def _savefig(name: str, stem: str = '') -> None:
    prefix = f'{stem}_' if stem else ''
    base = get_plots_path() / f'{prefix}{name}'
    plt.savefig(base.with_suffix('.pdf'), dpi=400)
    plt.savefig(base.with_suffix('.png'), dpi=400)

COLOR_TEXT     = "#1D51AC"
COLOR_DNA_BPE  = "#AB2617"
COLOR_DNA_KMER = COLOR_DNA_BPE

def init():
    sns.set_style('white')
    sns.set_context('paper', font_scale=1.2)

    plt.rcParams.update({
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "out",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
    })

def order_fisher(fisher: dict):
    x, y = ['embeddings'], [fisher['embeddings']]
    x.append('encoder')
    y.append(fisher['encoder'])
    x.append('head')
    y.append(fisher['head'])
    return x, np.array(y)

def order_fisher_full(fisher: dict):
    x, y = ['embeddings'], [fisher['embeddings']]

    for i in range(len(fisher) - 2):
        k = f'encoder.{i}'
        v = fisher[k]
        x.append(k)
        y.append(v)

    x.append('head')
    y.append(fisher['head'])

    return x, np.array(y)

def plot_fisher_information(text, dna_bpe, dna_kmer, stem: str = ''):
    print('plotting fisher information')
    print(text)
    print(dna_bpe)
    print(dna_kmer)

    xlabels, y_text = order_fisher(text)
    _, y_dna_bpe = order_fisher(dna_bpe)
    _, y_dna_kmer = order_fisher(dna_kmer)

    y_pos = np.arange(0, len(xlabels))

    y_text /= y_text.sum()
    y_dna_bpe /= y_dna_bpe.sum()
    y_dna_kmer /= y_dna_kmer.sum()

    init()

    fig, ax = plt.subplots(1, 2, figsize=(7, 2), sharey=True)

    ax[0].barh(y_pos, y_text, height=0.6, color=COLOR_TEXT, label='Text', linewidth=0.5, edgecolor='black')
    ax[0].legend()
    ax[0].set_xlim((0.0, 1.0))
    ax[0].set_xticks([0.0, 0.5, 1.0])

    bar_height = 0.4
    offset = bar_height / 2

    ax[1].barh(y_pos + offset, y_dna_bpe, height=bar_height, color=COLOR_DNA_BPE, label='DNA (BPE)', linewidth=0.5, edgecolor='black')
    ax[1].barh(y_pos - offset, y_dna_kmer, height=bar_height, color=COLOR_DNA_KMER, label=r'DNA ($k$-mer)', hatch='///', linewidth=0.5, edgecolor='black')
    ax[1].legend()
    ax[1].set_xlim((0.0, 1.0))
    ax[1].set_xticks([0.0, 0.5, 1.0])

    fig.text(0.45, 0.05, 'Normalized Fisher Information', va='center', rotation='horizontal', fontsize=10)

    def format_y(x, pos):
        label = xlabels[pos]
        if not isinstance(label, str):
            return label
        if '.' in label:
            label = label.replace('.', '-')
        label = label[0].capitalize() + label[1:]
        if label == 'Encoder':
            return 'Transformer Layers'
        return label

    ax[0].set_yticks(y_pos)
    ax[0].yaxis.set_major_formatter(FuncFormatter(format_y))
    ax[1].set_yticks(y_pos)

    ax[0].tick_params(axis='x', which='both', length=0)
    ax[1].tick_params(axis='x', which='both', length=0)

    ax[0].margins(y=0.2)
    ax[1].margins(y=0.2)

    fig.canvas.draw()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    _savefig('fisher', stem)
    plt.close()

def plot_full_fisher_information(text, dna_bpe, dna_kmer, stem: str = ''):
    N = len(text)
    assert N == len(dna_bpe) == len(dna_kmer)

    init()

    fig, ax = plt.subplots(N, 2, figsize=(7, 7.5), sharey=True)

    fig.subplots_adjust(
        top=0.99,     # smaller -> less space at top
        bottom=0.18,  # larger -> more space at bottom
        hspace=0.05,
        wspace=0.03,
        right=0.99,
        left=0.05,
    )

    for model in range(N):
        i = model

        xlabels, y_text = order_fisher_full(text[model])
        _, y_dna_bpe = order_fisher_full(dna_bpe[model])
        _, y_dna_kmer = order_fisher_full(dna_kmer[model])

        y_pos = np.arange(0, len(xlabels))

        y_text /= y_text.sum()
        y_dna_bpe /= y_dna_bpe.sum()
        y_dna_kmer /= y_dna_kmer.sum()

        width = 0.4

        def format_y(c, pos):
            label = xlabels[pos]
            label = label[0].capitalize() + label[1:]
            label = label.replace('.', ' ')
            return label

        kmer = r'$k$-mer'

        ax[i, 0].bar(y_pos, y_text, color=COLOR_TEXT, label=f'Text #{i+1}', linewidth=0.5, edgecolor='black')

        ax[i, 1].bar(y_pos - width/2, y_dna_bpe, width, color=COLOR_DNA_BPE, label=f'DNA (BPE) #{i+1}', edgecolor='black', linewidth=0.5)
        ax[i, 1].bar(y_pos + width/2, y_dna_kmer, width, color=COLOR_DNA_KMER, label=f'DNA ({kmer}) #{i+1}', hatch='///', edgecolor='black', linewidth=0.5)

        ax[i, 0].set_xticks(y_pos)
        ax[i, 0].set_xticklabels(xlabels)
        ax[i, 1].set_xticks(y_pos)
        ax[i, 1].set_xticklabels(xlabels)
        
        if i != N - 1:
            ax[i, 0].tick_params(axis='y', left=False, labelleft=False)
            ax[i, 1].tick_params(axis='y', left=False, labelleft=False)
            ax[i, 0].set_xticklabels([])
            ax[i, 1].set_xticklabels([])
        else:
            ax[i, 0].set_yticks([0.0, 0.5, 1.0])

            ax[i, 0].tick_params(axis='y', left=True, labelleft=True)
            ax[i, 1].tick_params(axis='y', left=False, labelleft=False)

            ax[i, 0].xaxis.set_major_formatter(FuncFormatter(format_y))
            ax[i, 1].xaxis.set_major_formatter(FuncFormatter(format_y))

        ax[i, 0].legend()
        ax[i, 1].legend()
        ax[i, 0].tick_params(axis='x', labelrotation=90)
        ax[i, 1].tick_params(axis='x', labelrotation=90)

        ax[i, 0].margins(x=0.02)
        ax[i, 1].margins(x=0.02)

    fig.canvas.draw()

    _savefig('full_fisher', stem)
    plt.close()

def plot_average_distribution(mean_dist_text, mean_dist_dna_bpe, mean_dist_dna_kmer, stem: str = ''):
    sns.set_style('white')
    sns.set_context('paper', font_scale=1.2)

    plt.rcParams.update({
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.labelsize": 10,         
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    fig, ax = plt.subplots(3, 1, figsize=(7, 5))
    #fig.patch.set_edgecolor('black')
    #fig.patch.set_linewidth(1)

    ax[0].bar(list(range(1, len(mean_dist_text)+1)), mean_dist_text, color=COLOR_TEXT, label='Text', width=1.0)
    ax[0].legend()
    ax[0].set_xticks([1, 5, 10])
    ax[0].set_yticks([0.0, 0.5, 1.0])
    ax[0].set_ylim((0.0, 1.0))

    ax[1].bar(list(range(1, len(mean_dist_dna_bpe)+1)), mean_dist_dna_bpe, color=COLOR_DNA_BPE, label='DNA (bpe)', width=1.0)
    ax[1].legend()
    ax[1].set_ylim((0.0, 0.04))
    ax[1].set_yticks([0.0, 0.1, 0.2])
    ax[1].set_xticks([1, 25, 50])

    ax[2].bar(list(range(1, len(mean_dist_dna_kmer)+1)), mean_dist_dna_kmer, color=COLOR_DNA_KMER, label=r'DNA ($k$-mer)', width=1.0)
    ax[2].set_ylim((0.0, 0.02))
    ax[2].set_yticks([0.0, 0.1, 0.2])
    ax[2].set_xticks([1, 25, 50])

    ax[2].set_xlabel('Token Rank')
    ax[2].legend()

    fig.text(0.02, 0.5, 'Average Softmax Probability', va='center', rotation='vertical')

    ax[0].margins(x=0.02)
    ax[1].margins(x=0.02)

    plt.tight_layout()
    plt.subplots_adjust(left=0.12)
    _savefig('dist', stem)
    plt.close()

def plot_jensen_shannon(text_js, dna_bpe_js, dna_kmer_js, stem: str = ''):
    sns.set_style('white')
    sns.set_context('paper', font_scale=1.2)

    plt.rcParams.update({
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.labelsize": 10,         
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    fig = plt.figure(figsize=(7, 2.5))

    p_values = list(reversed(list(text_js.keys())))
    text_js = list(reversed(list(text_js.values())))
    dna_bpe_js = list(reversed(list(dna_bpe_js.values())))
    dna_kmer_js = list(reversed(list(dna_kmer_js.values())))

    plt.plot(p_values, text_js, label='Text', color=COLOR_TEXT, linewidth=1.8)
    plt.plot(p_values, dna_bpe_js, label='DNA (BPE)', color=COLOR_DNA_BPE, linewidth=1.8)
    plt.plot(p_values, dna_kmer_js, label=r'DNA ($k$-mer)', color=COLOR_DNA_KMER, linewidth=1.8, linestyle='--')
    plt.gca().invert_xaxis()
    plt.ylim((0.0, max(max(dna_bpe_js), max(dna_kmer_js), max(text_js))+0.05))
    plt.ylabel('Jensen-Shannon Distance')
    plt.xlabel(r'Top-$p$ Mass Kept')
    plt.legend()

    plt.xticks([1.0, 0.5, 0.25, min(p_values)])
    plt.yticks([0.1, 0.5])
    for ax in fig.get_axes(): ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x*100)}%'))

    plt.margins(x=0, y=0)
    plt.tight_layout()
    _savefig('js', stem)
    plt.close()

def plot_attention_entropies(text, dna_bpe, dna_kmer, stem: str = ''):
    fig, ax = plt.subplots(1, figsize=(7, 7))

    x = list(sorted(list(text.keys())))

    ytext     = [text[l].item() for l in x]
    ydna_bpe  = [dna_bpe[l].item() for l in x]
    ydna_kmer = [dna_kmer[l].item() for l in x]

    minval = min([min(ytext), min(ydna_bpe), min(ydna_kmer)]) - 0.1
    maxval = max([max(ytext), max(ydna_bpe), max(ydna_kmer)]) + 0.1

    ax.set_ylim((minval, maxval))

    ax.plot(x, ytext, color=COLOR_TEXT, label='Text')
    ax.scatter(x, ytext, color=COLOR_TEXT)

    ax.plot(x, ydna_bpe, color=COLOR_DNA_BPE, label='DNA (BPE)')
    ax.scatter(x, ydna_bpe, color=COLOR_DNA_BPE)

    ax.plot(x, ydna_kmer, color=COLOR_DNA_BPE, linestyle='--', label='DNA (k-mer)')
    ax.scatter(x, ydna_kmer, color=COLOR_DNA_BPE)

    ax.set_xlabel('Transformer Layer')
    ax.set_ylabel('Mean Entropy')

    ax.legend()

    fig.tight_layout()
    _savefig('attn_entropy', stem)

def plot_attention_scores(text, dna_bpe, dna_kmer, stem: str = ''):
    fig, axes = plt.subplots(3, figsize=(10, 22))

    minval = 0.0
    maxval = max([text.max(), dna_bpe.max(), dna_kmer.max()])+0.1

    for ax, matrix, title in zip(axes, [text, dna_bpe, dna_kmer],
        ['Text', "DNA (BPE)", "DNA (KMER)"]):
    
        sns.heatmap(
            matrix, 
            annot=True, 
            fmt='.2f', 
            cmap='viridis', 
            vmin=minval, vmax=maxval,
            ax=ax,
            xticklabels=[f"L{(j+1)}" for j in range(matrix.shape[1])], 
            yticklabels=[f"L{(i+1)}" for i in range(matrix.shape[0])],
        )

        ax.set_title(title)

    #fig.suptitle('Mutual Information (MI) Between Attention Scores')
    fig.tight_layout()
    _savefig('attn', stem)

def plot_mi_matrix_full(mim_text, mim_dna_bpe, mim_dna_kmer, num_models, num_layers, model_names=None, stem: str = ''):
    if model_names is None:
        model_names = [f"M{i+1}" for i in range(num_models)]

    matrices = [mim_text, mim_dna_bpe, mim_dna_kmer]
    titles = ['Text', 'DNA BPE', 'DNA k-mer']

    tick_positions = [l * num_models + m for l in range(num_layers) for m in range(num_models)]
    tick_labels = [f"{model_names[m]}L{l+1}" for l in range(num_layers) for m in range(num_models)]

    # Shared color scale across all 3 matrices
    vmin = min(m.min() for m in matrices)

    vmax = 0.0
    for i in range(mim_text.shape[0]):
        for j in range(mim_text.shape[1]):
            if i == j: continue
            if mim_text[i, j] > vmax: vmax = mim_text[i, j]

    vmax += 0.1

    fig, axes = plt.subplots(3, 1, figsize=(12, 28))

    for ax, matrix, title in zip(axes, matrices, titles):
        im = ax.imshow(matrix, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label='MI Score')

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=7)

        for l in range(1, num_layers):
            pos = l * num_models - 0.5
            ax.axhline(pos, color='white', linewidth=0.8, linestyle='--')
            ax.axvline(pos, color='white', linewidth=0.8, linestyle='--')

        ax.set_title(f'MI Matrix — {title}')

    plt.tight_layout()
    _savefig('mi_matrix_full', stem)
    plt.close()
    #plt.show()