import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties


def fonts(folder_path, s=10, m=12):
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


def getSamplesData(dataPath, n5st, n10St, nIc, nKc):
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
        loss = dataframe['G" in Pa'].to_numpy()
        delta = dataframe['tan(δ) in -'].to_numpy()

        # Identifying segments in the data
        seg2, seg3, seg5, seg6 = (
            dataframe.index[dataframe['SegIndex'] == seg].to_list()[0] for seg in ['2|1', '3|1', '5|1', '5|31'])

        # Slice segments
        segments = lambda arr: (arr[seg2:seg3], arr[seg5:seg6])  # Returns (constant segment, step segment)

        return {
            'freq': segments(freq),
            'storage': segments(elastic),
            'loss': segments(loss),
            'delta': segments(delta)
        }

    # Store data for each sample type
    samples = {'5_st': [], '10_st': [], 'ic': [], 'kc': []}

    # Determine sample types for each path
    sample_labels = ['5_st'] * n5st + ['10_st'] * n10St + ['ic'] * nIc + ['kc'] * nKc

    # Read data and categorize based on sample type
    for sample_type, path in zip(sample_labels, dataPath):
        df = pd.read_excel(path)
        segments = getSegments(df)
        samples[sample_type].append(segments)

    # Initialize dictionaries to hold the results
    dict_freqSweeps = {}

    # Populate dictionaries with consolidated sample data
    for sample_type in samples:
        dict_freqSweeps[f'{sample_type}_freq'] = [s['freq'][0] for s in samples[sample_type]]
        dict_freqSweeps[f'{sample_type}_storage'] = [s['storage'][0] for s in samples[sample_type]]
        dict_freqSweeps[f'{sample_type}_loss'] = [s['loss'][0] for s in samples[sample_type]]
        dict_freqSweeps[f'{sample_type}_delta'] = [s['delta'][0] for s in samples[sample_type]]

        dict_freqSweeps[f'{sample_type}_freq_broken'] = [s['freq'][-1] for s in samples[sample_type]]
        dict_freqSweeps[f'{sample_type}_storage_broken'] = [s['storage'][-1] for s in samples[sample_type]]
        dict_freqSweeps[f'{sample_type}_loss_broken'] = [s['loss'][-1] for s in samples[sample_type]]
        dict_freqSweeps[f'{sample_type}_delta_broken'] = [s['delta'][-1] for s in samples[sample_type]]

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
        idSample = idSample + 1 if individualData else 'Mean'
        ax.set_title(axTitle, size=9, color='crimson')
        ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)

        ax.set_xlabel(f'{xLabel}')
        ax.set_xscale('log' if logScale else 'linear')
        ax.set_xlim(xLim)

        ax.set_ylabel(f'{yLabel}')
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
            color=curveColor, alpha=(0.9 - curve * 0.2) if individualData else 1.0,
            fmt=markerStyle, markersize=7, mfc=markerFColor, mec=markerEColor, mew=markerEWidth,
            capsize=3, lw=1, linestyle=lineStyle,
            label=f'{sampleName}_{idSample} | '
                  + "$\overline{G'} \\approx$" + f'{cteMean:.0f} ± {cteMeanErr:.0f} ' + '$Pa$',
            zorder=3)

        legendLabel()

    if individualData:
        for curve in range(nSamples):
            x, y, yerr = x[curve][::3], y[curve][::3], 0
            configPlot(idSample=curve)
    else:
        x = np.mean(x, axis=0)[:-2]
        yerr = np.std(y, axis=0)[:-2]
        y = np.mean(y, axis=0)[:-2]
        configPlot()


def main(dataPath):
    fonts('C:/Users/petrus.kirsten/AppData/Local/Microsoft/Windows/Fonts/')
    # samplesQuantities = list(samplesValues.keys())

    fileName = '0WSt_and_Car-ViscoelasticRecovery'
    dirSave = Path(*Path(filePath[0]).parts[:Path(filePath[0]).parts.index('data') + 1])

    plt.style.use('seaborn-v0_8-ticks')
    fig, axes = plt.subplots(figsize=(10, 7), facecolor='w', ncols=2, nrows=1)
    fig.suptitle(f'Frequency sweeps')

    yTitle, yLimits = f"Storage (G')" + f' and loss (G") moduli' + f' (Pa)', (7, 1.5 * 10 ** 4)
    xTitle, xLimits = f'Frequency (Hz)', (0.05, 150)
    ic5_color, st10_color, ic_color, kc_color = 'silver', 'sandybrown', 'hotpink', 'mediumturquoise'
    plotEachSample = False

    ic5_nSamples, st10_nSamples, ic_nSamples, kc_nSamples = 1, 2, 3, 2
    data = getSamplesData(dataPath, ic5_nSamples, st10_nSamples, ic_nSamples, kc_nSamples)

    (x_5st, gP_5st, gD_5st,
     x_10st, gP_10st, gD_10st,
     x_ic, gP_ic, gD_ic,
     x_kc, gP_kc, gD_kc) = [], [], [], [], [], [], [], [], [], [], [], []

    listBefore = {
        '5_st': ([], [], []),  # (x_5st, gP_5st, gD_5st)
        '10_st': ([], [], []),  # (x_10st, gP_10st, gD_10st)
        'ic': ([], [], []),  # (x_ic, gP_ic, gD_ic)
        'kc': ([], [], [])  # (x_kc, gP_kc, gD_kc)
    }
    listAfter = {
        '5_st': ([], [], []),  # (x_5st, gP_5st, gD_5st)
        '10_st': ([], [], []),  # (x_10st, gP_10st, gD_10st)
        'ic': ([], [], []),  # (x_ic, gP_ic, gD_ic)
        'kc': ([], [], [])  # (x_kc, gP_kc, gD_kc)
    }

    # Loop through each key in the listBefore dictionary
    for key, (x, gP, gD) in listBefore.items():
        x.append(data[f'{key}_freq'])
        gP.append(data[f'{key}_storage'])
        gD.append(data[f'{key}_loss'])

    for key, (x, gP, gD) in listAfter.items():
        x.append(data[f'{key}_freq_broken'])
        gP.append(data[f'{key}_storage_broken'])
        gD.append(data[f'{key}_loss_broken'])
    # TODO: iterar os valores para plotar antes e depois
    # After the loop, you can access the listBefore like:
    # x_5st, gP_5st, gD_5st = listBefore['5_st']
    # x_10st, gP_10st, gD_10st = listBefore['10_st']
    # x_ic, gP_ic, gD_ic = listBefore['ic']
    # x_kc, gP_kc, gD_kc = listBefore['kc']

    for ax in range(2):
        plotFreqSweeps(
            nSamples=ic5_nSamples,
            ax=axes[ax], x=x_5st, y=gP_5st,
            axTitle='', yLabel=yTitle, yLim=yLimits, xLabel=xTitle, xLim=xLimits,
            curveColor=ic5_color, markerStyle='o', markerFColor=ic5_color, markerEColor='k',
            sampleName=f'5_0WSt', individualData=plotEachSample, logScale=True)
        plotFreqSweeps(
            nSamples=ic5_nSamples,
            ax=axes[ax], x=x_5st, y=gD_5st,
            axTitle='', yLabel=yTitle, yLim=yLimits, xLabel=xTitle, xLim=xLimits,
            curveColor=ic5_color, markerStyle='o', markerFColor='w',
            markerEColor=ic5_color, markerEWidth=1.5, lineStyle='--',
            sampleName=f'', individualData=plotEachSample, logScale=True)

    # plotFreqSweeps(
    #     nSamples=st10_nSamples,
    #     ax=axes, x=x_10st, y=gP_10st,
    #     axTitle='', yLabel=yTitle, yLim=yLimits, xLabel=xTitle, xLim=xLimits,
    #     curveColor=st10_color, markerStyle='o', markerFColor=st10_color, markerEColor='k',
    #     sampleName=f'10_0WSt', individualData=plotEachSample, logScale=True)
    # plotFreqSweeps(
    #     nSamples=st10_nSamples,
    #     ax=axes, x=x_10st, y=gD_10st,
    #     axTitle='', yLabel=yTitle, yLim=yLimits, xLabel=xTitle, xLim=xLimits,
    #     curveColor=st10_color, markerStyle='o', markerFColor='w',
    #     markerEColor=st10_color, markerEWidth=1.5, lineStyle='--',
    #     sampleName=f'', individualData=plotEachSample, logScale=True)
    #
    # plotFreqSweeps(
    #     nSamples=ic_nSamples,
    #     ax=axes, x=x_ic, y=gP_ic,
    #     axTitle='', yLabel=yTitle, yLim=yLimits, xLabel=xTitle, xLim=xLimits,
    #     curveColor=ic_color, markerStyle='o', markerFColor=ic_color, markerEColor='k',
    #     sampleName=f'10_0WSt_iCar', individualData=plotEachSample, logScale=True)
    # plotFreqSweeps(
    #     nSamples=ic_nSamples,
    #     ax=axes, x=x_ic, y=gD_ic,
    #     axTitle='', yLabel=yTitle, yLim=yLimits, xLabel=xTitle, xLim=xLimits,
    #     curveColor=ic_color, markerStyle='o', markerFColor='w',
    #     markerEColor=ic_color, markerEWidth=1.5, lineStyle='--',
    #     sampleName=f'', individualData=plotEachSample, logScale=True)
    #
    # plotFreqSweeps(
    #     nSamples=kc_nSamples,
    #     ax=axes, x=x_kc, y=gP_kc,
    #     axTitle='', yLabel=yTitle, yLim=yLimits, xLabel=xTitle, xLim=xLimits,
    #     curveColor=kc_color, markerStyle='o', markerFColor=kc_color, markerEColor='k',
    #     sampleName=f'10_0WSt_kCar', individualData=plotEachSample, logScale=True)
    # plotFreqSweeps(
    #     nSamples=kc_nSamples,
    #     ax=axes, x=x_kc, y=gD_kc,
    #     axTitle='', yLabel=yTitle, yLim=yLimits, xLabel=xTitle, xLim=xLimits,
    #     curveColor=kc_color, markerStyle='o', markerFColor='w',
    #     markerEColor=kc_color, markerEWidth=1.5, lineStyle='--',
    #     sampleName=f'', individualData=plotEachSample, logScale=True)

    plt.subplots_adjust(wspace=0.0, top=0.890, bottom=0.14, left=0.05, right=0.95)
    # plt.tight_layout()
    plt.show()
    fig.savefig(f'{dirSave}' + f'\\{fileName}' + '.png', facecolor='w', dpi=600)

    print(f'\n\n· Chart saved at\n{dirSave}.')


if __name__ == '__main__':
    folderPath = "C:/Users/petrus.kirsten/PycharmProjects/RheometerPlots/data"
    # folderPath = "C:/Users/Petrus Kirsten/Documents/GitHub/RheometerPlots/data"
    filePath = [
        folderPath + "/091024/5_0WSt_kCar/5_0WSt_kCar-viscoRecoveryandFlow_1.xlsx",
        #
        folderPath + "/031024/10_0WSt/10_0WSt-viscRec_1.xlsx",
        folderPath + "/031024/10_0WSt/10_0WSt-viscRec_2.xlsx",
        #
        # folderPath + "10_0WSt_iCar/10_0WSt_iCar-viscoRecoveryandFlow_1.xlsx",
        folderPath + "/031024/10_0WSt_iCar/10_0WSt_iCar-viscoRecoveryandFlow_2.xlsx",
        folderPath + "/031024/10_0WSt_iCar/10_0WSt_iCar-viscoRecoveryandFlow_3.xlsx",
        folderPath + "/031024/10_0WSt_iCar/10_0WSt_iCar-viscoRecoveryandFlow_4.xlsx",
        #
        # folderPath + "/091024/10_0WSt_kCar/10_0WSt_kCar-viscoelasticRecovery-Flow_2a.xlsx",
        folderPath + "/091024/10_0WSt_kCar/10_0WSt_kCar-viscoelasticRecovery-Flow_3a.xlsx",
        folderPath + "/091024/10_0WSt_kCar/10_0WSt_kCar-viscoelasticRecovery-Flow_4a.xlsx",
    ]

    main(dataPath=filePath)
