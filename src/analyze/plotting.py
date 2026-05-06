'''Functions for plotting, quite messy.'''

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from src.utils import get_plots_path, DATA_TOKENIZER_PAIRS, run_key


def _savefig(name: str, stem: str = '') -> None:
    prefix = f'{stem}_' if stem else ''
    base = get_plots_path() / f'{prefix}{name}'
    plt.savefig(base.with_suffix('.pdf'), dpi=400)
    plt.savefig(base.with_suffix('.png'), dpi=400)

COLOR_TEXT     = "#1D51AC"
COLOR_DNA_BPE  = "#AB2617"
COLOR_DNA_KMER = COLOR_DNA_BPE

# Colors and legend for the dataset-type barplots
_DATASET_COLORS = {'wiki': '#56B4E9', 'OG2': '#E69F00', 'ncRNA': '#D55E00', 'cDNA': '#CC3311'}
_DATASET_LEGEND = {'#56B4E9': 'Wikipedia', '#E69F00': 'OpenGenome2', '#D55E00': 'ncRNA', '#CC3311': 'cDNA'}
# _DATASET_COLORS = {'wiki': '#56B4E9', 'OG2': '#E69F00', 'ncRNA': '#D55E00', 'cDNA': '#882200'}
# _DATASET_LEGEND = {'#56B4E9': 'Wikipedia', '#E69F00': 'OpenGenome2', '#D55E00': 'ncRNA', '#882200': 'cDNA'}
_TYPE_DISPLAY = {'OG2': 'OpenGenome2', 'ncRNA': 'ncRNA', 'cDNA': 'cDNA'}
_DATASET_LABELS = {
    run_key(data, tok, type): f'{"Text" if data == "text" else "DNA"} {"BPE" if tok == "bpe" else "k-mer"}{" " + _TYPE_DISPLAY.get(type, type) if type else ""}'
    for data, tok, type in DATA_TOKENIZER_PAIRS
}


def barplot(ax_or_fig, *lists, labels=None, colors=None, hatches=None, legend=None, title=None, ylabel="Value",
            legend_loc='best', legend_fontsize=9, ylim=None, tick_fontsize=10, ylabel_fontsize=10, figsize=(8, 5)):
    '''Bar plot with per-bar mean ± std annotations. Pass ax=None to create a new figure.'''
    means = [np.mean(lst) for lst in lists]
    stds  = [np.std(lst, ddof=1) if len(lst) > 1 else 0.0 for lst in lists]

    x = np.arange(len(lists))
    bar_labels = labels if labels is not None else [f"Group {i+1}" for i in range(len(lists))]
    bar_colors = colors if colors is not None else ["steelblue"] * len(lists)
    bar_hatches = hatches if hatches is not None else [''] * len(lists)

    if ax_or_fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = ax_or_fig

    bars = ax.bar(x, means, yerr=stds, capsize=6, color=bar_colors, edgecolor="black", alpha=0.8, hatch=bar_hatches)

    # Group labels: one label per unique first-word prefix, at the midpoint of its first two bars
    seen = {}
    for i, label in enumerate(bar_labels):
        prefix = label.split()[0]
        if prefix not in seen:
            seen[prefix] = []
        seen[prefix].append(i)

    tick_positions = [np.mean(positions) for positions in seen.values()]
    tick_texts     = list(seen.keys())
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_texts, rotation=0, fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelleft=False)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    if title:
        ax.set_title(title)

    if ylim is not None:
        ax.set_ylim(*ylim)

    if legend:
        handles = [Patch(facecolor=c, edgecolor="black", alpha=0.8, label=name)
                   for c, name in legend.items()]
        color_leg = ax.legend(handles=handles, loc=legend_loc, fontsize=legend_fontsize)
        ax.add_artist(color_leg)

    if hatches and len(set(bar_hatches)) > 1:
        hatch_handles = [
            Patch(facecolor='white', edgecolor='black', hatch='',  label='BPE'),
            Patch(facecolor='white', edgecolor='black', hatch='////', label=r'$k$-mer'),
        ]
        ax.legend(handles=hatch_handles, loc='upper right', fontsize=legend_fontsize)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.02 * max(means),
            f"{mean:.2f} ± {std:.2f}",
            ha="center", va="bottom", fontsize=10,
        )

    return fig, ax


def plot_kl_divergence(results: dict, stem: str = '') -> None:
    init()

    lists  = [results[run_key(data, tok, type)]['kl'] for data, tok, type in DATA_TOKENIZER_PAIRS]
    labels = [_DATASET_LABELS[run_key(data, tok, type)]   for data, tok, type in DATA_TOKENIZER_PAIRS]
    colors = [_DATASET_COLORS[type]                        for _, _, type in DATA_TOKENIZER_PAIRS]

    fig, ax = barplot(None, *lists, labels=labels, colors=colors,
                      legend=_DATASET_LEGEND, ylabel="KL divergence")

    plt.tight_layout()
    _savefig('kl_divergence', stem)
    plt.close()

def plot_entropy(results: dict, stem: str = '') -> None:
    init()

    lists  = [results[run_key(data, tok, type)]['entropy'] for data, tok, type in DATA_TOKENIZER_PAIRS]
    labels = [_DATASET_LABELS[run_key(data, tok, type)]    for data, tok, type in DATA_TOKENIZER_PAIRS]
    colors = [_DATASET_COLORS[type]                         for _, _, type in DATA_TOKENIZER_PAIRS]
    hatches = ['/' if tok == 'kmer' else '' for _, tok, _ in DATA_TOKENIZER_PAIRS]

    fig, ax = barplot(None, *lists, labels=labels, colors=colors, hatches=hatches,
                      legend=_DATASET_LEGEND, ylabel="Entropy (bits)",
                      legend_loc='upper left', legend_fontsize=7,
                      tick_fontsize=12, ylabel_fontsize=12, figsize=(8, 5))

    plt.tight_layout()
    _savefig('entropy', stem)
    plt.close()

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

def plot_fisher_information(results: dict, stem: str = ''):
    init()

    # 2x2 grid: one cell per dataset type; DNA cells show BPE + k-mer side by side
    _cells = [
        ('wiki',  'Wikipedia',   [('text', 'bpe', 'wiki')]),
        ('OG2',   'OpenGenome2', [('dna', 'bpe', 'OG2'),   ('dna', 'kmer', 'OG2')]),
        ('ncRNA', 'ncRNA',       [('dna', 'bpe', 'ncRNA'), ('dna', 'kmer', 'ncRNA')]),
        ('cDNA',  'cDNA',        [('dna', 'bpe', 'cDNA'),  ('dna', 'kmer', 'cDNA')]),
    ]

    _ylabels = ['Embeddings', 'Transformers', 'Head']
    def _fmt_label(val, _pos):
        i = int(round(val))
        return _ylabels[i] if 0 <= i < len(_ylabels) else ''

    def _per_model_stats(fisher_full):
        """Mean and std of normalized [embeddings, encoder, head] across 5 models."""
        all_y = []
        for model_dict in fisher_full.values():
            _, y_full = order_fisher_full(model_dict)
            y = np.array([y_full[0], y_full[1:-1].sum(), y_full[-1]])
            all_y.append(y / y.sum())
        arr = np.stack(all_y)
        return arr.mean(axis=0), arr.std(axis=0, ddof=1)

    y_pos = np.arange(3)
    fig, axes = plt.subplots(2, 2, figsize=(8, 4), sharey=True)

    for (type_, title, groups), ax in zip(_cells, axes.flat):
        color = _DATASET_COLORS[type_]
        stats = [_per_model_stats(results[run_key(d, t, ty)]['fisher_full']) for d, t, ty in groups]

        if len(stats) == 1:
            mean, std = stats[0]
            ax.barh(y_pos, mean, xerr=std, height=0.6, color=color,
                    edgecolor='black', linewidth=0.5, error_kw=dict(elinewidth=0.8, capsize=3))
        else:
            h = 0.4
            mean0, std0 = stats[0]
            mean1, std1 = stats[1]
            ax.barh(y_pos + h/2, mean0, xerr=std0, height=h, color=color,
                    edgecolor='black', linewidth=0.5, label='BPE',
                    error_kw=dict(elinewidth=0.8, capsize=3))
            ax.barh(y_pos - h/2, mean1, xerr=std1, height=h, color=color,
                    edgecolor='black', linewidth=0.5, hatch='///', label=r'$k$-mer',
                    error_kw=dict(elinewidth=0.8, capsize=3))
            ax.legend(fontsize=9)

        ax.set_title(title, fontsize=12)
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.tick_params(axis='x', which='both', length=0)
        ax.margins(y=0.2)
        ax.set_yticks(y_pos)
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_label))

    for ax in axes[1]:
        ax.set_xlabel('')

    fig.canvas.draw()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    left_pos  = axes[1, 0].get_position()
    right_pos = axes[1, 1].get_position()
    mid_x = (left_pos.x0 + right_pos.x1) / 2
    fig.text(mid_x, 0.02, 'Normalized Fisher Information', ha='center', fontsize=12)
    _savefig('fisher', stem)
    plt.close()

def plot_full_fisher_information(results: dict, stem: str = ''):
    # 2x2 grid; one cell per dataset type; bars show mean across 5 models
    _cells = [
        ('wiki',  'Wikipedia',   [('text', 'bpe', 'wiki')]),
        ('OG2',   'OpenGenome2', [('dna', 'bpe', 'OG2'),   ('dna', 'kmer', 'OG2')]),
        ('ncRNA', 'ncRNA',       [('dna', 'bpe', 'ncRNA'), ('dna', 'kmer', 'ncRNA')]),
        ('cDNA',  'cDNA',        [('dna', 'bpe', 'cDNA'),  ('dna', 'kmer', 'cDNA')]),
    ]

    def _mean_normalized(fisher_full_dict):
        all_y = []
        xlabels = None
        for f in fisher_full_dict.values():
            xl, y = order_fisher_full(f)
            if xlabels is None:
                xlabels = xl
            all_y.append(y / y.sum())
        return xlabels, np.stack(all_y).mean(axis=0)

    init()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    fig.subplots_adjust(top=0.95, bottom=0.18, hspace=0.2, wspace=0.12, left=0.07, right=0.99)

    for idx, ((type_, title, groups), ax) in enumerate(zip(_cells, axes.flat)):
        color = _DATASET_COLORS[type_]
        fisher_lists = [results[run_key(d, t, ty)]['fisher_full'] for d, t, ty in groups]

        xlabels, mean_y0 = _mean_normalized(fisher_lists[0])
        y_pos = np.arange(len(xlabels))
        row = idx // 2

        def _fmt_x(val, _pos):
            i = int(round(val))
            if 0 <= i < len(xlabels):
                lbl = xlabels[i][0].capitalize() + xlabels[i][1:]
                return lbl.replace('.', ' ')
            return ''

        if len(groups) == 1:
            ax.bar(y_pos, mean_y0, color=color, edgecolor='black', linewidth=0.5)
        else:
            _, mean_y1 = _mean_normalized(fisher_lists[1])
            w = 0.4
            ax.bar(y_pos - w/2, mean_y0, width=w, color=color, edgecolor='black', linewidth=0.5, label='BPE')
            ax.bar(y_pos + w/2, mean_y1, width=w, color=color, edgecolor='black', linewidth=0.5, hatch='///', label=r'$k$-mer')
            ax.legend(fontsize=9)

        ax.set_title(title, fontsize=12)
        ax.set_xticks(y_pos)
        ax.margins(x=0.02)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.tick_params(axis='y', labelsize=11)

        if row == 1:
            ax.xaxis.set_major_formatter(FuncFormatter(_fmt_x))
            ax.tick_params(axis='x', labelrotation=90, labelsize=10)
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis='x', length=0)

    for ax in axes[1]:
        ax.set_xlabel('Layer', fontsize=12)
    for ax in axes[:, 0]:
        ax.set_ylabel('Normalized Fisher Information', fontsize=12)

    fig.canvas.draw()
    _savefig('full_fisher', stem)
    plt.close()

def plot_average_distribution(results: dict, stem: str = ''):
    init()

    from matplotlib.gridspec import GridSpec
    n_rows = 4
    fig = plt.figure(figsize=(12, 2.2 * n_rows))
    gs = GridSpec(n_rows, 4, figure=fig, hspace=0.5, wspace=0.5)

    ax_slots = [
        fig.add_subplot(gs[0, 1:3]),
        fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4]),
        fig.add_subplot(gs[2, 0:2]), fig.add_subplot(gs[2, 2:4]),
        fig.add_subplot(gs[3, 0:2]), fig.add_subplot(gs[3, 2:4]),
    ]

    last_row_axes = ax_slots[5:]

    for i, (ax, (data, tok, type)) in enumerate(zip(ax_slots, DATA_TOKENIZER_PAIRS)):
        key = run_key(data, tok, type)
        n = 10 if data == 'text' else 50
        dist = results[key]['mean_dist'][:n]

        ax.bar(range(1, len(dist) + 1), dist, color=_DATASET_COLORS[type],
               width=1.0, edgecolor='black', linewidth=0.4)
        ax.set_title(_DATASET_LABELS[key], fontsize=12, pad=3)
        ax.margins(x=0.02)
        ax.set_xticks([1, n // 2, n])
        ax.tick_params(axis='both', labelsize=11)

        ymax = float(dist.max())
        decimals = 3 if i in (1, 2) else 2
        ax.set_yticks([0, round(ymax / 2, decimals), round(ymax, decimals)])

        if ax in last_row_axes:
            ax.set_xlabel('Token Rank', fontsize=12)

    fig.text(0.02, 0.5, 'Average Softmax Probability', va='center', rotation='vertical', fontsize=12)
    plt.subplots_adjust(left=0.10, top=0.95, bottom=0.08, right=0.97)
    _savefig('dist', stem)
    plt.close()

def plot_jensen_shannon(results: dict, stem: str = ''):
    init()

    fig, ax = plt.subplots(figsize=(7, 3))

    all_vals = []
    legend_handles = {}

    for data, tok, type_ in DATA_TOKENIZER_PAIRS:
        key = run_key(data, tok, type_)
        js = results[key]['js']

        p_values = list(reversed(list(js.keys())))
        values   = list(reversed(list(js.values())))
        all_vals.extend(values)

        color    = _DATASET_COLORS[type_]
        ls       = '--' if tok == 'kmer' else '-'
        label    = _DATASET_LABELS[key]

        ax.plot(p_values, values, color=color, linestyle=ls, linewidth=1.6, label=label)

        if color not in legend_handles:
            legend_handles[color] = Patch(facecolor=color, edgecolor=color,
                                          label=_DATASET_LEGEND[color])

    ax.invert_xaxis()
    ax.set_ylim(0.0, max(all_vals) + 0.05)
    ax.set_ylabel('Jensen-Shannon Distance')
    ax.set_xlabel(r'Top-$p$ mass kept')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x*100)}%'))
    ax.set_xticks([1.0, 0.5, 0.25, min(p_values)])
    ax.tick_params(axis='x', pad=6)
    ax.margins(x=0, y=0)

    color_legend = ax.legend(handles=list(legend_handles.values()), fontsize=8, loc='upper left')
    ax.add_artist(color_legend)

    solid = plt.Line2D([], [], color='gray', linestyle='-',  linewidth=1.4, label='BPE')
    dash  = plt.Line2D([], [], color='gray', linestyle='--', linewidth=1.4, label=r'$k$-mer')
    ax.legend(handles=[solid, dash], fontsize=8, loc='upper right')

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