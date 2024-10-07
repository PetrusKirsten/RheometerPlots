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


def getSamplesValues(dataPath, nSamplesWSt, nSamplesWStICar):
    """
    Processa dados de múltiplos caminhos fornecidos e armazena os resultados em um dicionário.

    Parâmetros:
    - dataPath (list): Lista de caminhos de arquivos .xlsx para serem processados.
    - nSamplesWSt (int): Número de amostras para o primeiro conjunto de dados.
    - nSamplesWStICar (int): Número de amostras para o segundo conjunto de dados.

    Retorno:
    - dictSamples (dict): Dicionário com as variáveis resultantes organizadas por diretório.
    """
    # Dicionário para armazenar resultados por caminho de arquivo
    dictSamples = {}

    # Arrays vazios para acumular os resultados de cada amostra
    rateWSt, stressWSt = np.array([]), np.array([])
    rateWStICar, stressWStICar = np.array([]), np.array([])

    # Iterar sobre cada caminho fornecido
    for sample, path in enumerate(dataPath):
        # Ler o DataFrame do arquivo
        df = pd.read_excel(path)

        # Processar os dados para calcular as métricas (rate e stress)
        if sample < nSamplesWSt:
            _, indRateWSt, indStressWSt = dataFlow(df)
            rateWSt = np.append(rateWSt, indRateWSt)
            stressWSt = np.append(stressWSt, indStressWSt)
        else:
            _, indRateWStICar, indStressWStICar = dataFlow(df)
            rateWStICar = np.append(rateWStICar, indRateWStICar)
            stressWStICar = np.append(stressWStICar, indStressWStICar)

    # Reshape das variáveis acumuladas
    (rateWSt,
     stressWSt) = (rateWSt.reshape(nSamplesWSt, rateWSt.shape[0] // nSamplesWSt),
                   stressWSt.reshape(nSamplesWSt, stressWSt.shape[0] // nSamplesWSt))
    (rateWStICar,
     stressWStICar) = (rateWStICar[1:].reshape(nSamplesWStICar, rateWStICar.shape[0] // nSamplesWStICar),
                       stressWStICar[1:].reshape(nSamplesWStICar, stressWStICar.shape[0] // nSamplesWStICar))

    # Armazenar os resultados no dicionário final
    (dictSamples['rateWSt'], dictSamples['stressWSt'],
     dictSamples['rateWStICar'], dictSamples['stressWStICar']) = (rateWSt, stressWSt,
                                                                  rateWStICar, stressWStICar)

    return dictSamples


def dataFlow(dataframe):
    (time,
     shearRate,
     shearStress) = (
        columnRead(dataframe, 't in s'),
        columnRead(dataframe, "GP in 1/s"),
        columnRead(dataframe, "Tau in Pa"))

    segInd1 = dataframe.index[dataframe['Seg'] == '3-1'].to_list()[0]
    segInd2 = dataframe.index[dataframe['Seg'] == '5-1'].to_list()[0]

    (timeSeg,
     shearRate,
     shearStress) = (
        time[segInd1: segInd2],
        shearRate[segInd1: segInd2],
        shearStress[segInd1: segInd2])

    return timeSeg - timeSeg[0], shearRate, shearStress


def plotFlow(
        ax, x, y,
        axTitle, textSize, curveColor,
        cteShear,  # in seg
        yFlowLim=(0, 600)

):
    """Plots the Flow Shearing Assay."""
    indexCteShear = np.where(x >= cteShear)[0][0]  # find the index where time >= final cte shear
    timeCteShear, timeDecShear = np.split(x, [indexCteShear + 1])

    ax.set_title(axTitle, size=9, color='crimson')
    ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)

    ax.set_xlabel('Shear rate ($s^{-1}$)')
    # ax.set_xticks([cteShear])
    # ax.set_xticklabels([f'{cteShear} s'])
    ax.set_xlim([x[-1] - 10, x[0] + 10])

    ax.set_ylabel('Shear stress (Pa)')
    ax.set_ylim(yFlowLim)
    # ax.set_yticks([])
    # ax.set_yscale('log')
    # Plot cte strain rate rectangle/line
    # textCoord, textLabel = ((np.median(timeCteShear), 5),
    #                         'Constant strain rate\n$\dot{γ}=300 \,\,\, s^{-1}$')
    # rectConfig = [(x[0] - cteShear, 0), 2 * cteShear, yFlowLim[-1] + 50]
    # ax.text(textCoord[0], textCoord[1], textLabel,
    #         horizontalalignment='center', verticalalignment='bottom',
    #         color='k', size=textSize)
    # rect = Rectangle(*rectConfig, linewidth=1, linestyle='--', edgecolor='grey', facecolor='w',
    #                  alpha=0.7, zorder=1)
    # ax.add_patch(rect)
    # Plot decreasing strain rate text
    # textCoord, textLabel = ((np.median(timeDecShear), 5),
    #                         'Step decrease of strain rate\n$300< \dot{γ} < 0.1 \,\,\, s^{-1}$')
    # ax.text(textCoord[0], textCoord[1], textLabel,
    #         horizontalalignment='center', verticalalignment='bottom',
    #         color='k', size=textSize)

    # Plot data
    ax.scatter(x, y, alpha=0.6,
               lw=.5, c=curveColor, edgecolor='k', s=30, marker='o',
               zorder=3)
    # legendLabel(ax)


def legendLabel(ax):
    """Applies consistent styling to legends in plots."""
    legend = ax.legend(frameon=True, framealpha=0.9, fancybox=False, scatterpoints=3)
    legend.get_frame().set_facecolor('w')
    legend.get_frame().set_edgecolor('whitesmoke')


# Main Plotting Configuration
def mainPlot(dataPath):
    samplesValues = getSamplesValues(dataPath, 2, 3)
    samplesQuantities = list(samplesValues.keys())

    # TODO: calcular médias, std dev, definir variávies e plotar
    
    sampleName = Path(filePath).stem
    dirSave = f'{Path(filePath).parent}' + f'{Path(filePath).stem}' + '.png'
    fonts('C:/Users/petrus.kirsten/AppData/Local/Microsoft/Windows/Fonts/')

    plt.style.use('seaborn-v0_8-ticks')
    fig, axes = plt.subplots(figsize=(8, 6), facecolor='w', ncols=1)
    fig.suptitle(f'Rheometry assay protocol to evaluate viscoelastic recovery')

    #  Plot 10% WSt
    plotFlow(
        axes, rateWSt, stressWSt,
        axTitle='', textSize=9.2, curveColor='orange',
        cteShear=300)

    #  Plot 10% WSt + iCar
    plotFlow(
        axes, rateWStICar, stressWStICar,
        axTitle='', textSize=9.2, curveColor='dodgerblue',
        cteShear=300)

    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.175, top=0.890, bottom=0.14, left=0.05, right=0.95)
    fig.savefig(dirSave, facecolor='w', dpi=600)
    plt.show()


# Run the main plotting function
filePath = [  # Pure WSt
    'C:/Users/petrus.kirsten/PycharmProjects/RheometerPlots/data/031024/10pct_0WSt/10pct_0WSt'
    '-RecoveryAndFlow_1.xlsx',
    'C:/Users/petrus.kirsten/PycharmProjects/RheometerPlots/data/031024/10pct_0WSt/10pct_0WSt'
    '-RecoveryAndFlow_2.xlsx',
    # WSt + iCar
    'C:/Users/petrus.kirsten/PycharmProjects/RheometerPlots/data/031024/10pct_0WSt_iCar/10pct_0WSt_iCar'
    '-RecoveryAndFlow_2.xlsx',
    'C:/Users/petrus.kirsten/PycharmProjects/RheometerPlots/data/031024/10pct_0WSt_iCar/10pct_0WSt_iCar'
    '-RecoveryAndFlow_3.xlsx',
    'C:/Users/petrus.kirsten/PycharmProjects/RheometerPlots/data/031024/10pct_0WSt_iCar/10pct_0WSt_iCar'
    '-RecoveryAndFlow_4.xlsx']  # -> CEBB PC
# filePath = ('C:/Users/petrus.kirsten/PycharmProjects/RheometerPlots/data/031024/10pct_0WSt/10pct_0WSt'
#             '-RecoveryAndFlow_2.xlsx')  # -> CEBB PC

# filePath = ('C:/Users/Petrus Kirsten/Documents/GitHub/RheometerPlots/data/031024/10pct_0WSt/10pct_0WSt'
#             '-RecoveryAndFlow_2.xlsx')  # personal PC

mainPlot(dataPath=filePath)
