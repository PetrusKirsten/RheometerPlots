import numpy as np
import pandas as pd
from pathlib import Path

from matplotlib.ticker import MultipleLocator
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


def getSamplesData(dataPath, n5st=0, n10St=0, nIc=0, n5Kc=0, n10Kc=1):
    """
    Reads multiple sample files and categorizes the data into 'cteRate' and 'stepsRate' dictionaries.
    """

    def getSegments(dataframe):
        """
        Extracts freq, shear rate, shear stress, and delta segments from the dataframe.
        Returns tuples of constant and step segments.
        """
        temperature = dataframe['T in °C'].to_numpy()
        stress = dataframe['τ in Pa'].to_numpy()
        viscosity = dataframe['η in mPas'].to_numpy()

        # Identifying segments in the data
        seg2, seg3 = (dataframe.index[dataframe['SegIndex'] == seg].to_list()[0] for seg in ['2|1', '2|10'])

        # Slice segments
        segments = lambda arr: (arr[seg2:seg3])  # Returns (constant segment, step segment)

        return {
            'temperature': segments(temperature),
            'stress': segments(stress),
            'viscosity': segments(viscosity)
        }

    # Store data for each sample type
    samples = {'10% WSt kCar': []}

    # Determine sample types for each path
    sample_labels = ['10% WSt kCar'] * n10Kc

    # Read data and categorize based on sample type
    for sample_type, path in zip(sample_labels, dataPath):
        df = pd.read_excel(path)
        segments = getSegments(df)
        samples[sample_type].append(segments)

    # Initialize dictionaries to hold the results
    dict_tempSweeps = {}

    # Populate dictionaries with consolidated sample data
    for sample_type in samples:
        dict_tempSweeps[f'{sample_type} temperature'] = [s['temperature'] for s in samples[sample_type]]
        dict_tempSweeps[f'{sample_type} stress'] = [s['stress'] for s in samples[sample_type]]
        dict_tempSweeps[f'{sample_type} viscosity'] = [s['viscosity'] for s in samples[sample_type]]

    return dict_tempSweeps


def plotTempSweeps(nSamples, sampleName,
                   ax, x, y,
                   axTitle, yLabel, yLim, xLabel, xLim, axisColor,
                   curveColor, markerStyle, markerFColor, markerEColor, markerEWidth=0.5,
                   lineStyle='', logScale=True):
    def legendLabel():
        """Applies consistent styling to legends in plots."""
        legend = ax.legend(fancybox=False, frameon=True, framealpha=0.9, fontsize=9)
        legend.get_frame().set_facecolor('w')
        legend.get_frame().set_edgecolor('whitesmoke')

    def configPlot():
        ax.set_title(axTitle, size=9, color='crimson')
        ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)

        ax.set_xlabel(f'{xLabel}')
        # ax.set_xscale('log' if logScale else 'linear')
        ax.set_xlim(xLim)
        ax.xaxis.set_minor_locator(MultipleLocator(5))

        ax.set_ylabel(f'{yLabel}', color=axisColor)
        ax.set_yscale('log' if logScale else 'linear')
        ax.set_ylim(yLim)
        ax.tick_params(axis='y', colors=axisColor, which='both')
        # ax.yaxis.set_minor_locator(MultipleLocator(TempSweeps.py)

        ax.errorbar(
            x[0], y[0], 0,
            color=curveColor, alpha=0.9,
            fmt=markerStyle, markersize=10, mfc=markerFColor, mec=markerEColor, mew=markerEWidth,
            capsize=3, lw=1, linestyle=lineStyle,
            label=f'{sampleName}',
            zorder=3)

    # x = np.mean(x, axis=0)
    # yerr = np.std(y, axis=0)
    # y = np.mean(y, axis=0)
    configPlot()


def main(dataPath):
    fonts('C:/Users/petrus.kirsten/AppData/Local/Microsoft/Windows/Fonts/')
    # samplesQuantities = list(samplesValues.keys())

    fileName = '10pct_0WSt_and_Car-TemperatureSweeps'
    dirSave = Path(*Path(filePath[0]).parts[:Path(filePath[0]).parts.index('data') + 1])

    plt.style.use('seaborn-v0_8-ticks')
    fig, axStress = plt.subplots(figsize=(10, 7), facecolor='w', ncols=1)
    axVisc = axStress.twinx()

    fig.suptitle(f'Temperature sweeps')

    yTitle, yLimits = f'Shear stress (Pa)', (10, 500)
    yTitleVisc, yLimitsVisc = f'Viscosity (mPa·s)', (100, 5 * 10 ** 4)
    xTitle, xLimits = f'Temperature (°C)', (0, 60)

    stressColor, viscosityColor = 'coral', 'blueviolet'

    st5_nSamples, st10_nSamples, ic_nSamples, kc_nSamples = 1, 2, 3, 2
    data = getSamplesData(dataPath)

    x_10kc, s_10kc, v_10kc = (
        data['10% WSt kCar temperature'],
        data['10% WSt kCar stress'],
        data['10% WSt kCar viscosity'])

    plotTempSweeps(
        nSamples=st5_nSamples,
        ax=axStress, x=x_10kc, y=s_10kc, axisColor=stressColor,
        axTitle='', yLabel=yTitle, yLim=yLimits, xLabel=xTitle, xLim=xLimits,
        curveColor=stressColor, markerStyle='o', markerFColor=stressColor, markerEColor='k',
        sampleName=f'')

    plotTempSweeps(
        nSamples=st5_nSamples,
        ax=axVisc, x=x_10kc, y=v_10kc, axisColor=viscosityColor,
        axTitle='', yLabel=yTitleVisc, yLim=yLimitsVisc, xLabel=xTitle, xLim=xLimits,
        curveColor=viscosityColor, markerStyle='o', markerFColor=viscosityColor, markerEColor='k',
        sampleName=f'')

    axVisc.spines['left'].set_color(stressColor)
    axVisc.spines['right'].set_color(viscosityColor)

    # plt.subplots_adjust(wspace=0.175, top=0.890, bottom=0.14, left=0.05, right=0.95)
    plt.tight_layout()
    plt.show()
    fig.savefig(f'{dirSave}' + f'\\{fileName}' + '.png', facecolor='w', dpi=600)

    print(f'\n\n· Chart saved at\n{dirSave}.')


if __name__ == '__main__':
    # folderPath = "C:/Users/petrus.kirsten/PycharmProjects/RheometerPlots/data"
    folderPath = "C:/Users/Petrus Kirsten/Documents/GitHub/RheometerPlots/data"
    filePath = [
        folderPath + "/091024/10_0WSt_kCar/10_0WSt_kCar-temperatureSweeps_1a.xlsx"
    ]

    main(dataPath=filePath)
