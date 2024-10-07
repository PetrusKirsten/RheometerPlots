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
    - dictCteRate (dict): Dicionário com as variáveis resultantes organizadas por diretório.
    """
    # Dicionário para armazenar resultados por caminho de arquivo
    dictCteRate, dictDecRate = {}, {}
    rateWStCte, rateWStDec, stressWStCte, stressWStDec = None, None, None, None
    rateWStICarCte, rateWStICarDec, stressWStICarCte, stressWStICarDec = None, None, None, None

    # Iterar sobre cada caminho fornecido
    for sample, path in enumerate(dataPath):
        # Ler o DataFrame do arquivo
        df = pd.read_excel(path)

        # Processar os dados para calcular as métricas (rate e stress)
        if sample < nSamplesWSt:
            _, indRateWSt, indStressWSt = dataFlow(df)

            rateWStCte, rateWStDec = (np.append(rateWStCte, indRateWSt[0]),
                                      np.append(rateWStDec, indRateWSt[1]))

            stressWStCte, stressWStDec = (np.append(stressWStCte, indStressWSt[0]),
                                          np.append(stressWStDec, indStressWSt[1]))

        else:
            _, indRateWStICar, indStressWStICar = dataFlow(df)

            rateWStICarCte, rateWStICarDec = (np.append(rateWStICarCte, indRateWStICar[0]),
                                              np.append(rateWStICarDec, indRateWStICar[1]))

            stressWStICarCte, stressWStICarDec = (np.append(stressWStICarCte, indStressWStICar[0]),
                                                  np.append(stressWStICarDec, indStressWStICar[1]))

    # Armazenar os resultados no dicionário final
    # Cte shear rate data
    #   Pure starch
    (dictCteRate['rateWStCte'],
     dictCteRate['stressWStCte']) = (rateWStCte[1:].reshape(nSamplesWSt,
                                                            rateWStCte.shape[0] // nSamplesWSt),
                                     stressWStCte[1:].reshape(nSamplesWSt,
                                                              stressWStCte.shape[0] // nSamplesWSt))
    #   Starch + iCar
    (dictCteRate['rateWStICarCte'],
     dictCteRate['stressWStICarCte']) = (rateWStICarCte[2:].reshape(nSamplesWStICar,
                                                                    rateWStICarCte[2:].shape[0] // nSamplesWStICar),
                                         stressWStICarCte[2:].reshape(nSamplesWStICar,
                                                                      stressWStICarCte[2:].shape[0] // nSamplesWStICar))

    # Decreasing shear rate data
    #   Pure starch
    (dictDecRate['rateWStDec'],
     dictDecRate['stressWStDec']) = (rateWStDec[1:].reshape(nSamplesWSt,
                                                            rateWStDec.shape[0] // nSamplesWSt),
                                     stressWStDec[1:].reshape(nSamplesWSt,
                                                              stressWStDec.shape[0] // nSamplesWSt))
    #   Starch + iCar
    (dictDecRate['rateWStICarDec'],
     dictDecRate['stressWStICarDec']) = (rateWStICarDec[1:].reshape(nSamplesWStICar,
                                                                    rateWStICarDec.shape[0] // nSamplesWStICar),
                                         stressWStICarDec[1:].reshape(nSamplesWStICar,
                                                                      stressWStICarDec.shape[0] // nSamplesWStICar))

    return dictCteRate, dictDecRate


def dataFlow(dataframe):
    time, shearRate, shearStress = (
        columnRead(dataframe, 't in s'),
        columnRead(dataframe, "GP in 1/s"),
        columnRead(dataframe, "Tau in Pa"))

    indexSeg3, indexSeg4, indexSeg5 = (dataframe.index[dataframe['Seg'] == '3-1'].to_list()[0],
                                       dataframe.index[dataframe['Seg'] == '4-1'].to_list()[0],
                                       dataframe.index[dataframe['Seg'] == '5-1'].to_list()[0])

    timeCte, timeDecrease = time[indexSeg3:indexSeg4], time[indexSeg4:indexSeg5]
    shearRateCte, shearRateDecrease = shearRate[indexSeg3:indexSeg4], shearRate[indexSeg4:indexSeg5]
    shearStressCte, shearStressDecrease = shearStress[indexSeg3:indexSeg4], shearStress[indexSeg4:indexSeg5]

    return ([timeCte - timeCte[0], timeDecrease - timeCte[0]],
            [shearRateCte, shearRateDecrease],
            [shearStressCte, shearStressDecrease])


def plotFlow(
        ax, x, y, yErr,
        axTitle, textSize, curveColor,
        sampleName,
        yFlowLim=(0, 600)

):
    """Plots the Flow Shearing Assay."""
    ax.set_title(axTitle, size=9, color='crimson')
    ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)

    ax.set_xlabel('Shear rate ($s^{-1}$)')
    # ax.set_xticks([cteShear])
    # ax.set_xticklabels([f'{cteShear} s'])
    ax.set_xlim([-25, +325])

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
    ax.errorbar(
        x, y, yerr=yErr, alpha=0.9, lw=.75, linestyle='-',
        color=curveColor, mec='k', fmt='o', markersize=6,
        label=sampleName, zorder=3, capsize=3)
    legendLabel(ax)


def legendLabel(ax):
    """Applies consistent styling to legends in plots."""
    legend = ax.legend(frameon=True, framealpha=0.9, fancybox=False, scatterpoints=3)
    legend.get_frame().set_facecolor('w')
    legend.get_frame().set_edgecolor('whitesmoke')


# Main Plotting Configuration
def mainPlot(dataPath):
    constantShear, decreasingShear = getSamplesValues(dataPath, 2, 3)
    # samplesQuantities = list(samplesValues.keys())

    fileName = '10pct_0WSt_and_iCar-stressXshear_031024'
    dirSave = f'{Path(filePath[0]).parent}' + f'\\{fileName}' + '.png'
    fonts('C:/Users/petrus.kirsten/AppData/Local/Microsoft/Windows/Fonts/')

    plt.style.use('seaborn-v0_8-ticks')
    fig, axes = plt.subplots(figsize=(8, 6), facecolor='w', ncols=1)
    fig.suptitle(f'Rheometry assay protocol to evaluate viscoelastic recovery')

    #  Plot 10% WSt
    xWSt, yWSt, yWStErr = (np.mean(decreasingShear['rateWStDec'], axis=0),
                           np.mean(decreasingShear['stressWStDec'], axis=0),
                           np.std(decreasingShear['stressWStDec'].astype(float), axis=0))
    plotFlow(
        axes, xWSt, yWSt, yWStErr,
        axTitle='', textSize=9.2, curveColor='orange',
        sampleName='10%_0WSt')

    #  Plot 10% WSt + iCar  # TODO: consertar o nro de pontos na amostra 10%_0WSt_iCar_3

    xWSt_iCar, yWSt_iCar, yWStErr_iCar = (np.mean(decreasingShear['rateWStICarDec'], axis=0),
                                          np.mean(decreasingShear['stressWStICarDec'], axis=0),
                                          np.std(decreasingShear['stressWStICarDec'].astype(float), axis=0))
    plotFlow(
        axes, xWSt_iCar, yWSt_iCar, yWStErr_iCar,
        axTitle='', textSize=9.2, curveColor='dodgerblue',
        sampleName='10%_0WSt_iCar')

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
