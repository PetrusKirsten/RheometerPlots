import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit


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


# TODO: fazer função para fitar modelo de HB
def powerLaw(sigma, k, n, sigmaZero):
    return sigmaZero + k * (sigma ** n)


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


def getSamplesValues(dataPath, nSt, nIc):
    """
    Processa dados de múltiplos caminhos fornecidos e armazena os resultados em um dicionário.

    Parâmetros:
    - dataPath (list): Lista de caminhos de arquivos .xlsx para serem processados.
    - nSamplesWSt (int): Número de amostras para o primeiro conjunto de dados.
    - nSamplesWStICar (int): Número de amostras para o segundo conjunto de dados.

    Retorno:
    - dict_cteRate (dict): Dicionário com as variáveis resultantes organizadas por diretório.
    """
    st_time_cte, ic_time_cte = None, None
    dict_cteRate, dict_stepsRate = {}, {}  # Dicionário para armazenar resultados por caminho de arquivo

    st_time, st_rateCte, st_rateSteps, st_stressCte, st_stressSteps = [], [], [], [], []
    ic_time, ic_rateCte, ic_rateSteps, ic_stressCte, ic_stressSteps = [], [], [], [], []

    vars_WSt = ('st_rateCte', 'st_rateSteps', 'st_stressCte', 'st_stressSteps')
    vars_iCar = ('ic_rateCte', 'ic_rateSteps', 'ic_stressCte', 'ic_stressSteps')

    for sample, path in enumerate(dataPath):
        df = pd.read_excel(path)

        if sample < nSt:
            st_time_i, st_rate_i, st_stress_i = dataFlow(df)

            st_time.append(st_time_i[0])
            st_rateCte.append(st_rate_i[0])
            st_rateSteps.append(st_rate_i[1])

            st_stressCte.append(st_stress_i[0])
            st_stressSteps.append(st_stress_i[1])

        else:
            ic_time_i, ic_rate_i, ic_stress_i = dataFlow(df)

            if sample == 3:
                ic_time.append(ic_time_i[0][::2])
                ic_rateCte.append(ic_rate_i[0][::2])
                ic_rateSteps.append(ic_rate_i[1][::2])

                ic_stressCte.append(ic_stress_i[0][::2])
                ic_stressSteps.append(ic_stress_i[1][::2])
            else:
                ic_time.append(ic_time_i[0])
                ic_rateCte.append(ic_rate_i[0])
                ic_rateSteps.append(ic_rate_i[1])

                ic_stressCte.append(ic_stress_i[0])
                ic_stressSteps.append(ic_stress_i[1])

    # Cte shear rate data
    #   Pure starch
    dict_cteRate[f'st_time'] = st_time
    dict_cteRate[f'{vars_WSt[0]}'] = st_rateCte
    dict_cteRate[f'{vars_WSt[2]}'] = st_stressCte

    dict_stepsRate[f'{vars_WSt[1]}'] = st_rateSteps
    dict_stepsRate[f'{vars_WSt[3]}'] = st_stressSteps

    #   Starch + iCar
    dict_cteRate[f'ic_time'] = ic_time

    dict_cteRate[f'{vars_iCar[0]}'] = ic_rateCte
    dict_cteRate[f'{vars_iCar[2]}'] = ic_stressCte

    dict_stepsRate[f'{vars_iCar[1]}'] = ic_rateSteps
    dict_stepsRate[f'{vars_iCar[3]}'] = ic_stressSteps

    return vars_WSt, vars_iCar, dict_cteRate, dict_stepsRate


def dataFlow(dataframe):
    time, shearRate, shearStress = (
        columnRead(dataframe, 't in s'),
        columnRead(dataframe, "GP in 1/s"),
        columnRead(dataframe, "Tau in Pa"))

    seg3, seg4, seg5 = (dataframe.index[dataframe['Seg'] == '3-1'].to_list()[0],
                        dataframe.index[dataframe['Seg'] == '4-1'].to_list()[0],
                        dataframe.index[dataframe['Seg'] == '5-1'].to_list()[0])

    tCte, tSteps = time[seg3:seg4], time[seg4:seg5]
    shearRate_cte, shearRate_steps = shearRate[seg3:seg4], shearRate[seg4:seg5]
    shearStress_cte, shearStress_steps = shearStress[seg3:seg4], shearStress[seg4:seg5]

    return ([tCte - tCte[0], tSteps - tCte[0]],
            [shearRate_cte, shearRate_steps],
            [shearStress_cte, shearStress_steps])


def plotFlow(
        ax, x, y, yErr,
        axTitle, curveColor, sampleName,
        logScale=False,
        fit=False,
        yFlowLim=(125, 475)):
    ax.set_title(axTitle, size=9, color='crimson')
    ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)
    ax.set_xlabel('Time (s)')
    ax.set_xscale('log' if logScale else 'linear')
    ax.set_xlim([-30, +850])
    # ax.set_xticks([cteShear])
    # ax.set_xticklabels([f'{cteShear} s'])
    ax.set_ylabel('Shear stress (Pa)')
    ax.set_yscale('log' if logScale else 'linear')
    ax.set_ylim(yFlowLim)
    # ax.set_yticks([])
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

    # ax.fill_between(
    #     (np.array(x, dtype=float)), (np.array(y, dtype=float) - yErr), (np.array(y, dtype=float) + yErr),
    #     color=curveColor, alpha=0.075,
    #     zorder=1)
    # label=f'{sampleName} - Desvio Padrão', zorder=1)
    if fit:
        ax.plot(
            x, y, color=curveColor, linestyle='-', linewidth=1,
            label=sampleName, zorder=2
        )
    else:
        ax.errorbar(
            x, y, yerr=0, color=curveColor, alpha=1,
            fmt='o', markersize=7, mec='k', mew=0.5,
            capsize=3, lw=1, linestyle='',  # ecolor='k'
            label=sampleName, zorder=3)

    legendLabel(ax)


def legendLabel(ax):
    """Applies consistent styling to legends in plots."""
    legend = ax.legend(frameon=True, framealpha=0.9, fancybox=False, scatterpoints=3)
    legend.get_frame().set_facecolor('w')
    legend.get_frame().set_edgecolor('whitesmoke')


def main(dataPath):
    fonts('C:/Users/petrus.kirsten/AppData/Local/Microsoft/Windows/Fonts/')
    # samplesQuantities = list(samplesValues.keys())

    fileName = '10pct_0WSt_and_iCar-stressXshear_031024'
    dirSave = f'{Path(filePath[0]).parent}' + f'\\{fileName}' + '.png'

    plt.style.use('seaborn-v0_8-ticks')
    fig, axes = plt.subplots(figsize=(10, 6), facecolor='w', ncols=1)
    fig.suptitle(f'Shear flow')

    st_names, ic_names, constantShear, stepsShear = getSamplesValues(dataPath, 2, 3)

    # y_st, y_stErr, y_ic, y_icErr = (
    #     np.mean(constantShear['st_stressCte'].astype(float), axis=0),
    #     np.std(constantShear['st_stressCte'].astype(float), axis=0),
    #     #
    #     np.mean(constantShear['ic_stressCte'].astype(float), axis=0),
    #     np.std(constantShear['ic_stressCte'].astype(float), axis=0))
    x_st = [np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48,
                      51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96,
                      99, 102, 105, 108, 111, 114, 117, 120, 123, 126, 129, 132, 135,
                      138, 141, 144, 147, 150, 153, 156, 159, 162, 165, 168, 171, 174,
                      177]),
            np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45,
                      48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105,
                      108, 111, 114, 117, 120, 123, 126, 129, 132, 135, 138, 141, 144,
                      147, 150, 153, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183,
                      186, 189, 192, 195, 198, 201, 204, 207, 210, 213, 216, 219, 222,
                      225, 228, 231, 234, 237, 240, 240, 250, 250, 250, 260, 260, 260,
                      260, 270, 270, 270, 280, 280, 280, 290, 290, 290, 290, 300])]
    y_st = [None, None]
    y_st[0], y_st[1], x_ic, y_ic = (
        constantShear['st_stressCte'][0],
        constantShear['st_stressCte'][1],
        #
        constantShear['ic_time'],
        constantShear['ic_stressCte'])

    for curve in range(1):
        params, _ = curve_fit(powerLaw, x_st[curve], y_st[curve], p0=(2, 1, 0))
        K, n, sigma_zero = params
        x_fit = np.linspace(0, 600, 600)
        y_fit = powerLaw(x_fit, K, n, sigma_zero)

        plotFlow(
            axes, x_fit.tolist(), y_fit.tolist(), yErr=0,
            axTitle='', curveColor='orange',
            sampleName=f'Fit 10%_0WSt_{curve + 1} | '
                       f'$K = {params[0]:.2f}, n = {params[1]:.2f}, \sigma_0 = {params[2]:.1f}$',
            fit=True)
        plotFlow(
            axes, x_st[curve].tolist(), y_st[curve], yErr=0,
            axTitle='', curveColor='orange',
            sampleName=f'10%_0WSt_{curve + 1}')

    for curve in range(2):
        params, _ = curve_fit(powerLaw, x_ic[curve], y_ic[curve])
        K, n, sigma_zero = params
        x_fit = np.linspace(0, 600, 600)
        y_fit = powerLaw(x_fit, K, n, sigma_zero)

        plotFlow(
            axes, x_fit.tolist(), y_fit.tolist(), yErr=0,
            axTitle='', curveColor='dodgerblue',
            sampleName=f'Fit 10%_0WSt_iCar_{curve + 1} | '
                       f'$K = {params[0]:.2f}, n = {params[1]:.2f}, \sigma_0 = {params[2]:.1f}$',
            fit=True)
        plotFlow(
            axes, x_ic[curve].tolist(), y_ic[curve], yErr=0,
            axTitle='', curveColor='dodgerblue',
            sampleName=f'10%_0WSt_iCar_{curve + 1}')

    # plt.subplots_adjust(wspace=0.175, top=0.890, bottom=0.14, left=0.05, right=0.95)
    plt.tight_layout()
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

main(dataPath=filePath)
