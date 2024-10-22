import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties


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


def getCteMean(values, tolerance=50):
    """
    :param values: to be analysed
    :param tolerance: the difference betweem two points data
    :return: the mean of the "cte" region and its indexes
    """
    diffs = np.abs(np.diff(values))  # Calcular as diferenças entre valores consecutivos

    constantRegions = diffs < tolerance  # Identificar regiões onde a diferença está abaixo do valor de tolerância

    # Encontrar os índices onde a condição é satisfeita
    iStart, iEnd = None, None
    lengthMax, lengthCurrent, currentStart = 0, 0, 0
    for i, is_constant in enumerate(constantRegions):
        if is_constant:
            if lengthCurrent == 0:
                currentStart = i
            lengthCurrent += 1
        else:
            if lengthCurrent > lengthMax:
                lengthMax = lengthCurrent
                iStart = currentStart
                iEnd = i
            lengthCurrent = 0

    if lengthCurrent > lengthMax:  # Checar se a última sequência é a maior constante
        iStart = currentStart
        iEnd = len(values) - 1

    if iStart is None or iEnd is None:  # Se nenhuma região constante foi encontrada
        return None, None, None

    mean = np.mean(values[iStart:iEnd + 1])  # Calcular a média da região constante encontrada
    stddev = np.std(values[iStart:iEnd + 1])  # Calcular a média da região constante encontrada

    return mean, stddev, iStart, iEnd


def getSamplesData(dataPath, nPowder, n5min, n25min):
    """
    Reads multiple sample files and categorizes the data into 'cteRate' and 'stepsRate' dictionaries.
    """

    def getSegments(dataframe):
        """
        Extracts freq, shear rate, shear stress, and delta segments from the dataframe.
        Returns tuples of constant and step segments.
        """
        freq = dataframe['f in Hz'].to_numpy()
        elastic = dataframe["G' in Pa"].to_numpy()
        # loss = dataframe['G" in Pa'].to_numpy()
        # delta = dataframe['tan(δ) in -'].to_numpy()

        # Identifying segments in the data
        seg2, seg3 = (dataframe.index[dataframe['SegIndex'] == seg].to_list()[0] for seg in ['2|1', '2|31'])

        # Slice segments
        segments = lambda arr: (arr[seg2:seg3])  # Returns (constant segment, step segment)

        return {
            'Frequency': segments(freq),
            'Storage': segments(elastic),
            # 'Loss': segments(loss)
            # 'tan(delta)': segments(delta)
        }

    # Store data for each sample type
    samples = {'Powder': [], 'Sol. at 5 min': [], 'Sol. at 25 min': []}

    # Determine sample types for each path
    sample_labels = ['Powder'] * nPowder + ['Sol. at 5 min'] * n5min + ['Sol. at 25 min'] * n25min

    # Read data and categorize based on sample type
    for sample_type, path in zip(sample_labels, dataPath):
        df = pd.read_excel(path)
        segments = getSegments(df)
        samples[sample_type].append(segments)

    # Initialize dictionaries to hold the results
    dict_freqSweeps = {}

    # Populate dictionaries with consolidated sample data
    for sample_type in samples:
        dict_freqSweeps[f'{sample_type} Frequency'] = [s['Frequency'] for s in samples[sample_type]]
        dict_freqSweeps[f'{sample_type} Storage'] = [s['Storage'] for s in samples[sample_type]]
        # dict_freqSweeps[f'{sample_type} Loss'] = [s['Loss'] for s in samples[sample_type]]
        # dict_freqSweeps[f'{sample_type} tan(delta)'] = [s['tan(delta)'] for s in samples[sample_type]]

    return dict_freqSweeps


def plotFreqSweeps(nSamples, sampleName,
                   ax, x, y,
                   axTitle, yLabel, yLim, xLabel, xLim,
                   curveColor, markerStyle, markerFColor, markerEColor, markerEWidth=0.5,
                   lineStyle='-', individualData=False, logScale=False):
    def legendLabel():
        """Applies consistent styling to legends in plots."""
        legend = ax.legend(fancybox=False, frameon=True, framealpha=0.9, fontsize=9)
        legend.get_frame().set_facecolor('w')
        legend.get_frame().set_edgecolor('whitesmoke')

    def configPlot(idSample=0):
        cteMean, cteMeanErr, indexStart_mean, indexEnd_mean = getCteMean(y)
        dotCteMean = 'k'
        graphColor = 'darkgray'
        idSample = idSample + 1 if individualData else ''
        ax.set_title(axTitle, size=9, color='crimson')
        ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(1.2)
        ax.spines[['top', 'bottom', 'left', 'right']].set_color(graphColor)
        ax.tick_params(axis='both', which='both', length=0, labelcolor=graphColor)
        ax.grid(True, which='both', axis='both', linestyle=':', linewidth=1, color='darkgray', alpha=0.5)

        ax.set_xlabel(f'{xLabel}', color=graphColor)
        ax.set_xscale('log' if logScale else 'linear')
        ax.set_xlim(xLim)

        ax.set_ylabel(f'{yLabel}', color=graphColor)
        ax.set_yscale('log' if logScale else 'linear')
        ax.set_ylim(yLim)

        ax.errorbar(
            [x[indexStart_mean], x[indexEnd_mean]], [y[indexStart_mean], y[indexEnd_mean]], yerr=0,
            color=dotCteMean, alpha=0.75,
            fmt='.', markersize=4, mfc=dotCteMean, mec=dotCteMean, mew=1,
            capsize=0, lw=1, linestyle='',
            label=f'', zorder=4)

        ax.errorbar(
            x, y, yerr,
            color=curveColor, alpha=(0.9 - curve * 0.2) if individualData else .75,
            fmt='o', markersize=7, mfc=markerFColor, mec=markerEColor, mew=markerEWidth,
            capsize=3, lw=1, linestyle='',
            label=f'{sampleName} | '
                  + "$\overline{G'} \\approx$" + f'{cteMean:.0f} ± {cteMeanErr:.0f} ' + '$Pa$',
            zorder=3)

        legendLabel()

    if individualData:
        for curve in range(nSamples):
            x, y, yerr = x[curve][::3], y[curve][::3], 0
            configPlot(idSample=curve)
    else:
        x = np.mean(x, axis=0)
        yerr = np.std(y, axis=0)
        y = np.mean(y, axis=0)
        configPlot()


def main(dataPath):
    fonts('C:/Users/petrus.kirsten/AppData/Local/Microsoft/Windows/Fonts/')
    # samplesQuantities = list(samplesValues.keys())

    fileName = '10pct_0WSt_iCar_CaCl2-FrequencySweepsToCLmethod'
    dirSave = Path(*Path(filePath[0]).parts[:Path(filePath[0]).parts.index('data') + 1])

    plt.style.use('seaborn-v0_8-ticks')
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='w', ncols=1)
    fig.suptitle(f'Frequency sweeps')

    yTitle, yLimits = f"Storage (G') (Pa)", (4*10**2, 2*10**4)
    # yTitle, yLimits = f"Storage (G')" + f' and loss (G") moduli' + f' (Pa)', (7, 1.5*10**4)
    xTitle, xLimits = f'Frequency (Hz)', (0.07, 130)
    powderColor, sol5minColor, sol25minColor = 'mediumaquamarine', 'hotpink', 'dodgerblue'

    data = getSamplesData(dataPath, 2, 2, 2)

    (f_powder, gP_powder,  # gD_powder,
     f_5min, gP_5min,  # gD_5min,
     f_25min, gP_25min) = (  # gD_25min) = (
        data['Powder Frequency'],
        data['Powder Storage'],
        # data['Powder Loss'],
        #
        data['Sol. at 5 min Frequency'],
        data['Sol. at 5 min Storage'],
        # data['Sol. at 5 min Loss'],
        #
        data['Sol. at 25 min Frequency'],
        data['Sol. at 25 min Storage'])
    # data['Sol. at 25 min Loss'])

    plotFreqSweeps(
        nSamples=2,
        ax=ax, x=f_5min, y=gP_5min,
        axTitle='', yLabel=yTitle, yLim=yLimits, xLabel=xTitle, xLim=xLimits,
        curveColor=sol5minColor, markerStyle='o', markerFColor=sol5minColor, markerEColor='k',
        sampleName=f'Add CL at 5 min', logScale=True)
    # plotFreqSweeps(
    #     nSamples=2,
    #     ax=ax, x=f_5min, y=gD_5min,
    #     axTitle='', yLabel=yTitle, yLim=yLimits, xLabel=xTitle, xLim=xLimits,
    #     curveColor=sol5minColor, markerStyle='o', markerFColor='w',
    #     markerEColor=sol5minColor, markerEWidth=1.5, lineStyle='--',
    #     sampleName=f'', logScale=True)

    plotFreqSweeps(
        nSamples=2,
        ax=ax, x=f_25min, y=gP_25min,
        axTitle='', yLabel=yTitle, yLim=yLimits, xLabel=xTitle, xLim=xLimits,
        curveColor=sol25minColor, markerStyle='o', markerFColor=sol25minColor, markerEColor='k',
        sampleName=f'Add CL at 25 min', logScale=True)
    # plotFreqSweeps(
    #     nSamples=2,
    #     ax=ax, x=f_25min, y=gD_25min,
    #     axTitle='', yLabel=yTitle, yLim=yLimits, xLabel=xTitle, xLim=xLimits,
    #     curveColor=sol25minColor, markerStyle='o', markerFColor='w',
    #     markerEColor=sol25minColor, markerEWidth=1.5, lineStyle='--',
    #     sampleName=f'', logScale=True)

    plotFreqSweeps(
        nSamples=2,
        ax=ax, x=f_powder, y=gP_powder,
        axTitle='', yLabel=yTitle, yLim=yLimits, xLabel=xTitle, xLim=xLimits,
        curveColor=powderColor, markerStyle='o', markerFColor=powderColor, markerEColor='k',
        sampleName=f'Add powder to dry polymers', logScale=True)
    # plotFreqSweeps(
    #     nSamples=2,
    #     ax=ax, x=f_powder, y=gD_powder,
    #     axTitle='', yLabel=yTitle, yLim=yLimits, xLabel=xTitle, xLim=xLimits,
    #     curveColor=powderColor, markerStyle='o', markerFColor='w',
    #     markerEColor=powderColor, markerEWidth=1.5, lineStyle='--',
    #     sampleName=f'', logScale=True)

    # plt.subplots_adjust(wspace=0.175, top=0.890, bottom=0.14, left=0.05, right=0.95)
    plt.tight_layout()
    plt.show()
    fig.savefig(f'{dirSave}' + f'\\{fileName}' + '.png', facecolor='w', dpi=600)

    print(f'\n\n· Chart saved at\n{dirSave}.')


if __name__ == '__main__':
    folderPath = "C:/Users/petrus.kirsten/PycharmProjects/RheometerPlots/data/161024/"
    # folderPath = "C:/Users/Petrus Kirsten/Documents/GitHub/RheometerPlots/data/161024/"
    filePath = [
        folderPath + "freqSweeps_dryPowderCL.xlsx",
        folderPath + "freqSweeps_dryPowderCL_2.xlsx",
        #
        folderPath + "freqSweeps_SolAt5minCL.xlsx",
        folderPath + "freqSweeps_SolAt5minCL_2.xlsx",
        #
        folderPath + "freqSweeps_SolAt25minCL.xlsx",
        folderPath + "freqSweeps_SolAt25minCL_2.xlsx"
    ]

    main(dataPath=filePath)
