from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import get_plots_path

def plot_average_distribution(mean_dist_text, mean_dist_dna):
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
    fig, ax = plt.subplots(2, 1, figsize=(7, 4))
    fig.patch.set_edgecolor('black')
    fig.patch.set_linewidth(1)

    ax[0].bar(list(range(1, len(mean_dist_text)+1)), mean_dist_text, color='royalblue', label='Text', edgecolor='black', linewidth=0.5)
    ax[0].legend()
    ax[0].set_xticks([1, 5, 10])
    ax[0].set_yticks([0.0, 0.4, 0.8])
    ax[0].set_ylim((0.0, 0.8))

    ax[1].bar(list(range(1, len(mean_dist_dna)+1)), mean_dist_dna, color='firebrick', label='DNA', edgecolor='black', linewidth=0.5)
    ax[1].set_ylim((0.0, 0.02))
    ax[1].set_yticks([0.0, 0.01, 0.02])
    ax[1].set_xticks([1, 25, 50])

    ax[1].set_xlabel('Token Rank')
    ax[1].legend()

    fig.text(0.02, 0.5, 'Average Softmax Probability', va='center', rotation='vertical')

    ax[0].margins(x=0.01)
    ax[1].margins(x=0.01)

    plt.tight_layout()
    plt.subplots_adjust(left=0.12)
    plt.savefig(get_plots_path() / 'dist.pdf')
    plt.close()

def plot_jensen_shannon(text_js, dna_js):
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

    fig = plt.figure(figsize=(7, 3))
    fig.patch.set_edgecolor('black')
    fig.patch.set_linewidth(1)

    p_values = list(reversed(list(text_js.keys())))
    text_js = list(reversed(list(text_js.values())))
    dna_js = list(reversed(list(dna_js.values())))
    plt.plot(p_values, text_js, label='Text', color='royalblue', linewidth=2)
    plt.plot(p_values, dna_js, label='DNA', color='firebrick', linewidth=2)
    plt.gca().invert_xaxis()
    plt.ylim((0.0, max(dna_js)+0.05))
    plt.ylabel('Jensen-Shannon Distance')
    plt.xlabel(r'Top-$p$ Mass Kept')
    plt.legend()

    plt.xticks([1.0, 0.5, 0.25, min(p_values)])
    plt.yticks([0.1, 0.4])
    for ax in fig.get_axes(): ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x*100)}%'))

    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.savefig(get_plots_path() / 'js.pdf')
    plt.close()
