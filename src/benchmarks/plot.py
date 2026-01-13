import numpy as np
import matplotlib.pyplot as plt
from src.common import get_plots_path

gb_markov = {
    'human_enhancers_cohn': {'acc': 0.7035118019573978, 'mcc': 0.40803127798331346, 'f1': 0.7135706340378198},
    'human_enhancers_ensembl': {'acc': 0.8327090732967388, 'mcc': 0.665494786148514, 'f1': 0.8314299658369937},
    'human_nontata_promoters': {'acc': 0.993579809608147, 'mcc': 0.987071683483425, 'f1': 0.9929748062015504},
    'human_ocr_ensembl': {'acc': 0.6503776607919433, 'mcc': 0.32339698593319993, 'f1': 0.5716489063376332},
}

gb_linear= {
    'human_enhancers_cohn': {'acc': 0.7238054116292458, 'mcc': 0.44772660327733227, 'f1': 0.7206289125054594},
    'human_enhancers_ensembl': {'acc': 0.85947368421053, 'mcc': 0.7201062721669863, 'f1': 0.8496772210267446},
    'human_nontata_promoters': {'acc': 0.9945760460482621, 'mcc': 0.9890674533121302, 'f1': 0.994052676295667},
    'human_ocr_ensembl': {'acc': 0.7055676356145572, 'mcc': 0.4119387823272367, 'f1': 0.7144799267541547},
}

gb_best = {
    'human_enhancers_cohn': {'acc': 0.747},
    'human_enhancers_ensembl': {'acc': 0.90},
    'human_nontata_promoters': {'acc': 0.957},
    'human_ocr_ensembl': {'acc': 0.828},
}

gb_worst = {
    'human_enhancers_cohn': {'acc': 0.729},
    'human_enhancers_ensembl': {'acc': 0.849},
    'human_nontata_promoters': {'acc': 0.945},
    'human_ocr_ensembl': {'acc': 0.783},
}

nt_markov = {
    'promoter_all': {'acc': 0.8503787878787878, 'mcc': 0.702579491499999, 'f1': 0.8447937131630648},
    'promoter_tata': {'acc': 0.8584905660377359, 'mcc': 0.7190318509781667, 'f1': 0.8636363636363636},
    'promoter_no_tata': {'acc': 0.8637026239067055, 'mcc': 0.731112968993511, 'f1': 0.8564850345356869},
    'enhancers': {'acc': 0.714, 'mcc': 0.44470990219777395, 'f1': 0.737454100367197},
    'H2AFZ': {'acc': 0.6743333333333333, 'mcc': 0.34873718712583396, 'f1': 0.6755230820325473},
    'H3K27ac': {'acc': 0.6732673267326733, 'mcc': 0.3488030432842367, 'f1': 0.6908665105386417},
    'H3K27me3': {'acc': 0.7306666666666667, 'mcc': 0.4733017023559579, 'f1': 0.756185878092939},
    'H3K36me3': {'acc': 0.6753333333333333, 'mcc': 0.3755514892815131, 'f1': 0.7262507026419337},
    'H3K4me1': {'acc': 0.668, 'mcc': 0.3729352424447698, 'f1': 0.7284623773173392},
    'H3K4me2': {'acc': 0.706267539756782, 'mcc': 0.41274389512274345, 'f1': 0.7108655616942909},
    'H3K4me3': {'acc': 0.7667525773195877, 'mcc': 0.5352161700184748, 'f1': 0.7570469798657719},
    'H3K9ac': {'acc': 0.7051792828685259, 'mcc': 0.4103878797386975, 'f1': 0.7034068136272545},
    'H3K9me3': {'acc': 0.62, 'mcc': 0.2400538312546489, 'f1': 0.6239813736903376},
}

nt_linear = {
    'promoter_all': {'acc': 0.8529040404040404, 'mcc': 0.7064982872921184, 'f1': 0.8495803744351195},
    'promoter_tata': {'acc': 0.8537735849056604, 'mcc': 0.7113879915305903, 'f1': 0.8609865470852018},
    'promoter_no_tata': {'acc': 0.8615160349854227, 'mcc': 0.7249602782061627, 'f1': 0.8562783661119516},
    'enhancers': {'acc': 0.7363333333333333, 'mcc': 0.4832793381948632, 'f1': 0.7513360578434455},
    'H2AFZ': {'acc': 0.7043333333333334, 'mcc': 0.4122373113116927, 'f1': 0.7208057916273214},
    'H3K27ac': {'acc': 0.7017326732673267, 'mcc': 0.40386640316536176, 'f1': 0.7082324455205811},
    'H3K27me3': {'acc': 0.7416666666666667, 'mcc': 0.4902899398837985, 'f1': 0.7601361807489941},
    'H3K36me3': {'acc': 0.7146666666666667, 'mcc': 0.43870547206729793, 'f1': 0.7427884615384616},
    'H3K4me1': {'acc': 0.7043333333333334, 'mcc': 0.4149435516935836, 'f1': 0.7293256026853829},
    'H3K4me2': {'acc': 0.725444340505145, 'mcc': 0.46429161795498997, 'f1': 0.7547012118679481},
    'H3K4me3': {'acc': 0.7835051546391752, 'mcc': 0.5748849751233094, 'f1': 0.7640449438202247},
    'H3K9ac': {'acc': 0.7529880478087649, 'mcc': 0.5098798230108779, 'f1': 0.7367303609341825},
    'H3K9me3': {'acc': 0.6105882352941177, 'mcc': 0.22451318476892576, 'f1': 0.574002574002574},
}

nt_worst = {
    'promoter_all': {'mcc': 0.67},
    'promoter_tata': {'mcc': 0.65},
    'promoter_no_tata': {'mcc': 0.72},
    'enhancers': {'mcc': 0.47},
    'H2AFZ': {'mcc': 0.46},
    'H3K27ac': { 'mcc': 0.21,},
    'H3K27me3': { 'mcc': 0.54, },
    'H3K36me3': { 'mcc': 0.54,},
    'H3K4me1': { 'mcc': 0.42, },
    'H3K4me2': { 'mcc': 0.43,},
    'H3K4me3': { 'mcc': 0.45, },
    'H3K9ac': { 'mcc': 0.3, },
    'H3K9me3': { 'mcc': 0.23,},
}

nt_best = {
    'promoter_all': {'mcc': 0.76},
    'promoter_tata': {'mcc': 0.94},
    'promoter_no_tata': {'mcc': 0.77},
    'enhancers': {'mcc': 0.61},
    'H2AFZ': {'mcc': 0.52},
    'H3K27ac': { 'mcc': 0.52,},
    'H3K27me3': { 'mcc': 0.6, },
    'H3K36me3': { 'mcc': 0.64,},
    'H3K4me1': { 'mcc': 0.5, },
    'H3K4me2': { 'mcc': 0.63,},
    'H3K4me3': { 'mcc': 0.63, },
    'H3K9ac': { 'mcc': 0.59, },
    'H3K9me3': { 'mcc': 0.48,},
}

def plot_nt():
    path = get_plots_path()

    labels = list(nt_markov.keys())
    x = np.arange(len(labels))

    n_series = 4
    width = 0.8 / n_series  # total group width ~0.8

    fig, ax = plt.subplots(figsize=(20, 5))

    ax.bar(
        x - 1.5*width,
        [v['mcc'] for v in nt_markov.values()],
        width,
        label="markov chain",
        color='blue',
        edgecolor='black',
        linewidth=1,
    )
    ax.bar(
        x - 0.5*width,
        [v['mcc'] for v in nt_linear.values()],
        width,
        label="linear",
        color='purple',
        edgecolor='black',
        linewidth=1,
    )
    ax.bar(
        x + 0.5*width,
        [v['mcc'] for v in nt_worst.values()],
        width,
        label="worst GLM",
        color='red',
        edgecolor='black',
        linewidth=1,
        hatch='\\\\',
    )
    ax.bar(
        x + 1.5*width,
        [v['mcc'] for v in nt_best.values()],
        width,
        label="best GLM",
        color='green',
        edgecolor='black',
        linewidth=1,
        hatch='//',
    )

    ax.set_title('Performance on NT-Tasks (MCC)')
    ax.set_ylabel('MCC')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path / 'nt.png', dpi=500)

def plot_gb():
    path = get_plots_path()

    labels = list(gb_markov.keys())
    x = np.arange(len(labels))

    n_series = 4
    width = 0.5 / n_series

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        x - 1.5*width,
        [v['acc'] for v in gb_markov.values()],
        width,
        label="markov chain",
        color='blue',
        edgecolor='black',
        linewidth=1,
    )
    ax.bar(
        x - 0.5*width,
        [v['acc'] for v in gb_linear.values()],
        width,
        label="linear",
        color='green',
        edgecolor='black',
        linewidth=1,
    )
    ax.bar(
        x + 0.5*width,
        [v['acc'] for v in gb_worst.values()],
        width,
        label="worst GLM",
        color='red',
        edgecolor='black',
        linewidth=1,
        hatch='\\\\',
    )
    ax.bar(
        x + 1.5*width,
        [v['acc'] for v in gb_best.values()],
        width,
        label="best GLM",
        color='purple',
        edgecolor='black',
        linewidth=1,
        hatch='//',
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title('Performance on Genomic Benchmarks (Accuracy)')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path / 'gb.png', dpi=500)

if __name__ == '__main__':
    plot_nt()
    plot_gb()
