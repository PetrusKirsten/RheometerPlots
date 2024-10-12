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


def plotFreqSweeps(sampleName,
                   ax, x, yP, yD, yPerr, yDerr,
                   axTitle, yLabel, yLim, xLabel, xLim,
                   curveColor, markerStyle,
                   lineStyle='-', individualData=False, logScale=True):
    def legendLabel():
        """Applies consistent styling to legends in plots."""
        legend = ax.legend(fancybox=False, frameon=True, framealpha=0.9, fontsize=9)
        legend.get_frame().set_facecolor('w')
        legend.get_frame().set_edgecolor('whitesmoke')

    def configPlot(idSample=0):
        meanStorage, storageMeanErr, indexStart_storage, indexEnd_storage = getCteMean(yP[0])
        dotCteMean = 'k'
        idSample = idSample + 1 if individualData else 'Mean'
        ax.set_title(axTitle, size=9, color='crimson')
        ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)

        ax.set_xlabel(f'{xLabel}')
        ax.set_xscale('log' if logScale else 'linear')
        ax.set_xlim(xLim)
        ax.grid(True, which='both', axis='y', linestyle='-', linewidth=0.5, color='silver', alpha=0.5)
        ax.set_ylabel(f'{yLabel}')
        ax.set_yscale('log' if logScale else 'linear')
        ax.set_ylim(yLim)

        # ax.errorbar(
        #     [x[indexStart_storage], x[indexEnd_storage]], [yP[indexStart_storage], yP[indexEnd_storage]], yerr=0,
        #     color=dotCteMean, alpha=0.75,
        #     fmt='.', markersize=4, mfc=dotCteMean, mec=dotCteMean, mew=1,
        #     capsize=0, lw=1, linestyle='',
        #     label=f'', zorder=4)

        ax.errorbar(
            x[0], yP[0], yPerr[0],
            color=curveColor, alpha=0.85,
            fmt=markerStyle, markersize=7, mfc=curveColor, mec='k', mew=0.5,
            capsize=3, lw=0.75, linestyle=lineStyle,
            label=f'{sampleName}_{idSample} | '
                  + "$\overline{G'} \\approx$" + f'{meanStorage:.0f} ± {storageMeanErr:.0f} ' + '$Pa$',
            zorder=3)

        ax.errorbar(
            x[0], yD[0], yDerr[0],
            color=curveColor, alpha=0.85,
            fmt=markerStyle, markersize=7, mfc='w', mec=curveColor, mew=0.75,
            capsize=3, lw=0.75, linestyle=':',
            zorder=3)

        legendLabel()

    configPlot()


def main(dataPath):
    fonts('C:/Users/petrus.kirsten/AppData/Local/Microsoft/Windows/Fonts/')

    fileName = '0WSt_and_Car-ViscoelasticRecovery'
    dirSave = Path(*Path(filePath[0]).parts[:Path(filePath[0]).parts.index('data') + 1])

    plt.style.use('seaborn-v0_8-ticks')
    fig, axes = plt.subplots(figsize=(18, 7), facecolor='w', ncols=2, nrows=1)

    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_label_position('right')

    fig.suptitle(f'Viscoelastic recovery by frequency sweeps assay.')

    yTitle, yLimits = f"Storage (G')" + f' and loss (G") moduli' + f' (Pa)', (1, 2*10**5)
    xTitle, xLimits = f'Frequency (Hz)', (0.06, 200)

    ic5_nSamples, st10_nSamples, ic_nSamples, kc_nSamples = 1, 2, 3, 2
    nSamples = (ic5_nSamples, st10_nSamples, ic_nSamples, kc_nSamples)
    ic5_color, st10_color, ic_color, kc_color = 'grey', 'sandybrown', 'deepskyblue', 'hotpink'
    colors = (ic5_color, st10_color, ic_color, kc_color)

    data = getSamplesData(dataPath, *nSamples)

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

    for key, (x, gP, gD) in listBefore.items():
        x.append(data[f'{key}_freq'])
        gP.append(data[f'{key}_storage'])
        gD.append(data[f'{key}_loss'])

    for key, (x, gP, gD) in listAfter.items():
        x.append(data[f'{key}_freq_broken'])
        gP.append(data[f'{key}_storage_broken'])
        gD.append(data[f'{key}_loss_broken'])

    for k_a, k_b, c in zip(listAfter, listBefore, colors):

        plotFreqSweeps(
            ax=axes[0], x=np.mean(listBefore[k_a][0], axis=1),
            yP=np.mean(listBefore[k_a][1], axis=1), yD=np.mean(listBefore[k_a][2], axis=1),
            yPerr=np.std(listBefore[k_a][1], axis=1), yDerr=np.std(listBefore[k_a][2], axis=1),
            axTitle='Before breakage', yLabel=yTitle, yLim=yLimits, xLabel=xTitle, xLim=xLimits,
            curveColor=c, markerStyle='o',
            sampleName=k_a, logScale=True)

        plotFreqSweeps(
            ax=axes[1], x=np.mean(listAfter[k_a][0], axis=1),
            yP=np.mean(listAfter[k_a][1], axis=1), yD=np.mean(listAfter[k_a][2], axis=1),
            yPerr=np.std(listAfter[k_a][1], axis=1), yDerr=np.std(listAfter[k_a][2], axis=1),
            axTitle='After breakage', yLabel=yTitle, yLim=yLimits, xLabel=xTitle, xLim=xLimits,
            curveColor=c, markerStyle='o',
            sampleName=k_a, logScale=True)

    plt.subplots_adjust(wspace=0.0, top=0.93, bottom=0.1, left=0.05, right=0.95)
    # plt.tight_layout()
    plt.show()
    fig.savefig(f'{dirSave}' + f'\\{fileName}' + '.png', facecolor='w', dpi=600)

    print(f'\n\n· Chart saved at\n{dirSave}.')


if __name__ == '__main__':
    # folderPath = "C:/Users/petrus.kirsten/PycharmProjects/RheometerPlots/data"
    folderPath = "C:/Users/Petrus Kirsten/Documents/GitHub/RheometerPlots/data"
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
