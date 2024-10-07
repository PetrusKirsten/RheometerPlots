import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle


def fonts(folder_path, s=11, m=13):
    """Configures font properties for plots."""
    font_path = folder_path + 'HelveticaNeueThin.otf'
    helvetica_thin = FontProperties(fname=font_path)

    font_path = folder_path + 'HelveticaNeueLight.otf'
    helvetica_light = FontProperties(fname=font_path)

    font_path = folder_path + 'HelveticaNeueMedium.otf'
    helvetica_medium = FontProperties(fname=font_path)

    font_path = folder_path + 'HelveticaNeueBold.otf'
    helvetica_bold = FontProperties(fname=font_path)

    plt.rc('font', size=s)  # controls default text sizes
    plt.rc('axes', titlesize=s)  # fontsize of the axes title
    plt.rc('axes', labelsize=s)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=s)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=s)  # fontsize of the tick labels
    plt.rc('legend', fontsize=s)  # legend fontsize
    plt.rc('figure', titlesize=m)  # fontsize of the figure title


def constantMean(values, tolerance=100):
    """
    Encontra a região de valores praticamente constantes e calcula sua média.

    Parâmetros:
    - values (array-like): Array de valores numéricos a serem analisados.
    - tolerance (float): Tolerância para definir regiões constantes. Quanto menor, mais rigoroso.

    Retorno:
    - mean (float): Média da região constante encontrada.
    - start (int): Índice inicial da região constante.
    - end (int): Índice final da região constante.
    """
    # Calcular as diferenças entre valores consecutivos
    diffs = np.abs(np.diff(values))

    # Identificar regiões onde a diferença está abaixo do valor de tolerância
    constant_regions = diffs < tolerance

    # Encontrar os índices onde a condição é satisfeita
    indexStart, indexEnd = None, None
    max_length, current_length = 0, 0
    current_start = 0

    for i, is_constant in enumerate(constant_regions):
        if is_constant:
            if current_length == 0:
                current_start = i
            current_length += 1
        else:
            if current_length > max_length:
                max_length = current_length
                indexStart = current_start
                indexEnd = i
            current_length = 0

    # Checar se a última sequência é a maior constante
    if current_length > max_length:
        indexStart = current_start
        indexEnd = len(values) - 1

    # Se nenhuma região constante foi encontrada
    if indexStart is None or indexEnd is None:
        return None, None, None

    # Calcular a média da região constante encontrada
    mean = np.mean(values[indexStart:indexEnd + 1])

    return mean, indexStart, indexEnd


def columnRead(dataframe, key):
    """
    :param dataframe: the Pandas dataframe to read
    :param key: the column label/title
    :return: the values from column in a numpy array
    """
    return dataframe[key].to_numpy()


def dataFreqSweep(dataframe):
    (stress,
     gPrime,
     gDouble) = (columnRead(dataframe, 'Tau in Pa'),
                 columnRead(dataframe, "G' in Pa"),
                 columnRead(dataframe, "G'' in Pa"))

    return stress, gPrime, gDouble, constantMean(gPrime), constantMean(gDouble)


def plotModuli(
        ax, x, gP, gD, gPmean, gDmean, markerSize, axTitle, curveColor,
        textConfig, textLabel, textCoord, rectConfig,
        yModLimits=(1, 50000),
        showRest=False
):
    """Plots the Oscillatory Frequency Sweep Assay."""
    ax.set_title(axTitle, size=10, color='crimson')
    ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)

    ax.set_xlabel('Shear stress (Pa)')
    # ax.set_xticks([])
    ax.set_xlim([x[0], x[-1] + 100])
    ax.set_xscale('log')

    ax.set_ylabel('Storage and loss moduli (Pa)')
    ax.set_ylim(yModLimits)
    ax.set_yscale('log')

    ax.plot(x[1:], gP[1:], lw=1, alpha=0.85,
            c=curveColor, marker='v', markersize=markerSize / 7, mew=0.5, mec='k',
            label=f"G' ~ {gPmean[0]:.0f} Pa", zorder=2)
    ax.plot(x[1:], gD[1:], lw=1, alpha=0.8,
            c=curveColor, marker='^', markersize=markerSize / 7, mew=0.75, mec=curveColor, mfc='w',
            label=f'G" ~ {gDmean[0]:.0f} Pa', zorder=2)

    if showRest:
        ax.text(textCoord[0], textCoord[1], s=textLabel, **textConfig)
        rect = Rectangle(*rectConfig, linewidth=1, edgecolor='k', facecolor='whitesmoke', alpha=0.75, zorder=1)
        ax.add_patch(rect)

    # ax.grid(ls='--', color='mediumaquamarine', alpha=0.5, zorder=0)
    legendLabel(ax)


def legendLabel(ax):
    """Applies consistent styling to legends in plots."""
    legend = ax.legend(frameon=True, framealpha=0.9, fancybox=False, scatterpoints=3)
    legend.get_frame().set_facecolor('w')
    legend.get_frame().set_edgecolor('whitesmoke')


# Main Plotting Configuration
def mainPlot(dataPath, color):
    df = pd.read_excel(dataPath)
    sampleName = Path(filePath).stem
    dirSave = f'{Path(filePath).parent}' + f'{Path(filePath).stem}' + '.png'

    fonts('C:/Users/petrus.kirsten/AppData/Local/Microsoft/Windows/Fonts/')
    markerSize = 50
    plt.style.use('seaborn-v0_8-ticks')
    fig, axes = plt.subplots(figsize=(8, 6), facecolor='w', ncols=1)
    fig.suptitle(f'Rheometry assay protocol to evaluate linear viscoelastic region | {sampleName}')

    # Common parameters
    text_coord = (0, 100)
    text_label = 'Rest and\nSet $T=37\,^oC$\nfor $180\,\,s$'
    text_properties = {'horizontalalignment': 'center',
                       'verticalalignment': 'top',
                       'color': 'k',
                       'size': 9.2}
    rect_properties = [(0, 0), 10, 100000]

    # Plot 1: Oscillatory Frequency Sweep Assay
    timeAx1, storage, loss, storageMean, lossMean = dataFreqSweep(df)
    plotModuli(
        axes, timeAx1, storage, loss, storageMean, lossMean, markerSize,
        axTitle='', curveColor=color,
        textConfig=text_properties, textLabel=text_label, textCoord=text_coord, rectConfig=rect_properties,
        showRest=False)

    # plt.subplots_adjust(wspace=0.175, top=0.890, bottom=0.14, left=0.05, right=0.95)
    plt.tight_layout()
    fig.savefig(dirSave, facecolor='w', dpi=600)
    plt.show()


# Run the main plotting function

filePath = ('C:/Users/petrus.kirsten/PycharmProjects/RheometerPlots/data/031024/10pct_0WSt/'
            '5pct_0WSt-StressSweep.xlsx')  # -> CEBB PC
# filePath = ('C:/Users/petrus.kirsten/PycharmProjects/RheometerPlots/data/031024/10pct_0WSt/10pct_0WSt'
#             '-RecoveryAndFlow_2.xlsx')  # -> CEBB PC

# filePath = ('C:/Users/Petrus Kirsten/Documents/GitHub/RheometerPlots/data/031024/10pct_0WSt/10pct_0WSt'
#             '-RecoveryAndFlow_2.xlsx')  # personal PC

mainPlot(dataPath=filePath, color='orange')
