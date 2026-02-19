'''Functions for plotting, quite messy.'''

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from src.utils import get_plots_path

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

def plot_fisher_information(text, dna_bpe, dna_kmer):
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
    plt.savefig(get_plots_path() / 'fisher.pdf')
    plt.close()

def plot_full_fisher_information(text, dna_bpe, dna_kmer):
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

    plt.savefig(get_plots_path() / 'full_fisher.pdf')
    plt.close()

def plot_average_distribution(mean_dist_text, mean_dist_dna_bpe, mean_dist_dna_kmer):
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
    ax[1].set_yticks([0.0, 0.01, 0.02, 0.04])
    ax[1].set_xticks([1, 25, 50])

    ax[2].bar(list(range(1, len(mean_dist_dna_kmer)+1)), mean_dist_dna_kmer, color=COLOR_DNA_KMER, label=r'DNA ($k$-mer)', width=1.0)
    ax[2].set_ylim((0.0, 0.02))
    ax[2].set_yticks([0.0, 0.01, 0.02])
    ax[2].set_xticks([1, 25, 50])

    ax[2].set_xlabel('Token Rank')
    ax[2].legend()

    fig.text(0.02, 0.5, 'Average Softmax Probability', va='center', rotation='vertical')

    ax[0].margins(x=0.02)
    ax[1].margins(x=0.02)

    plt.tight_layout()
    plt.subplots_adjust(left=0.12)
    plt.savefig(get_plots_path() / 'dist.pdf')
    plt.close()

def plot_jensen_shannon(text_js, dna_bpe_js, dna_kmer_js):
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
    plt.savefig(get_plots_path() / 'js.pdf')
    plt.close()
