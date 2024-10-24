import numpy as np
import pandas as pd
from pathlib import Path

from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt, ticker
from matplotlib.ticker import MultipleLocator
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


def fitLinear(strain, slope, intercept):
    return slope*strain + intercept


def arraySplit(xArr, yArr, startValue, endValue):
    startIndex, endIndex = np.where(xArr >= startValue)[0][0], np.where(xArr <= endValue)[0][-1]

    return xArr[startIndex:endIndex], yArr[startIndex:endIndex]


def getSamplesInfos(
        # quantites
        stCL_n, st_kcCL_n, st_icCL_n,
        kc_n, kcCL_n,
        # colors
        stCL_color, st_kcCL_color, st_icCL_color,
        kc_color, kcCL_color
):
    number_samples = [
        stCL_n, st_kcCL_n, st_icCL_n,
        kc_n, kcCL_n]

    colors_samples = [
        stCL_color,  st_kcCL_color, st_icCL_color,
        kc_color, kcCL_color]

    return number_samples, colors_samples


def getSamplesData(
        dataPath,
        number_samples
):
    def getSegments(dataframe):
        time = dataframe['t in s'].to_numpy()
        height = dataframe['h in mm'].to_numpy()
        force = dataframe['Fn in N'].to_numpy()

        seg_init, seg_end = (  # Identifying the job segments in the lists
            dataframe.index[dataframe['SegIndex'] == seg].to_list()[0] for seg in ['62|1', '62|98'])

        segmentsBreakage = lambda arr: (arr[seg_init:seg_end])  # Slice segments

        return {
            'time to break': segmentsBreakage(time) - segmentsBreakage(time)[0],
            'height to break': (1 - segmentsBreakage(height) / segmentsBreakage(height).max())*100,
            'force to break': segmentsBreakage(force)}

    samples = {
        '10%-0St/CL': [], '10%-0St + kCar/CL': [], '10%-0St + iCar/CL': [],
        'kCar': [], 'kCar/CL': []
    }
    sample_keys = list(samples.keys())
    sample_labels = (
            [sample_keys[0]] * number_samples[0] +
            [sample_keys[1]] * number_samples[1] +
            [sample_keys[2]] * number_samples[2] +
            [sample_keys[3]] * number_samples[3] +
            [sample_keys[4]] * number_samples[4]
    )
    for sample_type, path in zip(sample_labels, dataPath):  # Read data and categorize based on sample type
        df = pd.read_excel(path)
        segments = getSegments(df)
        samples[sample_type].append(segments)

    dict_data = {}  # Initialize dictionaries to hold the results

    for sample_type in samples:  # Populate dictionaries with consolidated sample data
        dict_data[f'{sample_type} time to break'] = [s['time to break'] for s in samples[sample_type]]
        dict_data[f'{sample_type} height to break'] = [s['height to break'] for s in samples[sample_type]]
        dict_data[f'{sample_type} force to break'] = [s['force to break'] for s in samples[sample_type]]

    return dict_data, sample_keys


def plotCompression(sampleName,
                    ax, x, y,
                    axTitle, yLabel, yLim, xLabel, xLim, axisColor,
                    curveColor, markerStyle, markerFColor, markerEColor, markerEWidth=0.5,
                    strain=False, linearFitting=False, logScale=False):
    def legendLabel():
        """Applies consistent styling to legends in plots."""
        legend = ax.legend(loc=4, fancybox=False, frameon=False, framealpha=0.9, fontsize=10)
        legend.get_frame().set_facecolor('w')
        legend.get_frame().set_edgecolor('whitesmoke')

    def configPlot():
        ax.set_title(axTitle, size=9, color='crimson')
        ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)
        # ax.grid(True, which='both', axis='x', linestyle='-', linewidth=0.5, color='lightsteelblue', alpha=0.5)

        ax.set_xlabel(f'{xLabel}')
        # ax.set_xscale('log' if logScale else 'linear')
        ax.set_xlim(xLim)
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}%"))

        ax.set_ylabel(f'{yLabel}', color=axisColor)
        ax.set_yscale('log' if logScale else 'linear')
        ax.set_ylim(yLim)
        ax.tick_params(axis='y', colors=axisColor, which='both')
        ax.yaxis.set_major_locator(MultipleLocator(100))
        ax.yaxis.set_minor_locator(MultipleLocator(50))

    if strain:
        ax.plot(
            x, y,
            color=curveColor, alpha=0.8, lw=1.5, linestyle=':',
            label=f'{sampleName}', zorder=3)

    elif linearFitting:
        # configs to split the values at the linear elastic region
        startVal, endVal = 6, 16
        x_toFit, y_toFit = arraySplit(x, y, startVal, endVal)
        (slope, intercept), covariance = curve_fit(fitLinear, x_toFit, y_toFit)  # p0=(y_mean[0], y_mean[-1], 100))
        (slopeErr, interceptErr) = np.sqrt(np.diag(covariance))

        xFit = np.linspace(startVal, endVal, 100)
        yFit = fitLinear(xFit, slope, intercept)
        ax.plot(
            xFit, yFit,
            color='crimson', alpha=0.8, lw=1.25, linestyle='-',
            label=f'Linear fitting', zorder=4)

        # show text and rectangle at the linear region
        textLabel, textCoord = 'Linear elastic region', (xFit[-1] + 2, np.median(yFit))
        textConfig = {'horizontalalignment': 'left', 'verticalalignment': 'top', 'color': 'crimson', 'size': 10}
        rectConfig = [(xFit[0], 0), xFit[-1] - xFit[0], yFit[-1] + 2]
        ax.text(textCoord[0], textCoord[1], s=textLabel, **textConfig)

        textLabel, textCoord = (
            f'YM = ${slope * 100:.1f}$ $±$ ${slopeErr * 100:.1f}$ $Pa$', (xFit[-1] + 1, np.median(yFit) - 20))
        textConfig = {'horizontalalignment': 'left', 'verticalalignment': 'top', 'color': 'k', 'size': 9}
        ax.text(textCoord[0], textCoord[1], s=textLabel, **textConfig)

        textLabel, textCoord = f'{startVal}%', (xFit[0] + 0.5, 3)
        textConfig = {'horizontalalignment': 'left', 'verticalalignment': 'bottom', 'color': 'k', 'size': 8}
        ax.text(textCoord[0], textCoord[1], s=textLabel, **textConfig, zorder=5)

        textLabel, textCoord = f'{endVal}%', (xFit[-1] - 0.5, 3)
        textConfig = {'horizontalalignment': 'right', 'verticalalignment': 'bottom', 'color': 'k', 'size': 8}
        ax.text(textCoord[0], textCoord[1], s=textLabel, **textConfig, zorder=5)

        rect = Rectangle(
            *rectConfig, linewidth=1, edgecolor='w', facecolor='crimson', alpha=0.1, zorder=1)
        ax.add_patch(rect)

    else:
        ax.errorbar(
            x, y, 0,
            color=curveColor, alpha=0.85,
            fmt=markerStyle, markersize=6, mfc=markerFColor, mec=markerEColor, mew=markerEWidth,
            capsize=0, lw=1, linestyle='',
            label=f'{sampleName}',
            zorder=3)
    legendLabel()
    configPlot()


def main(dataPath, fileName):
    fonts('C:/Users/petrus.kirsten/AppData/Local/Microsoft/Windows/Fonts/')
    plt.style.use('seaborn-v0_8-ticks')

    fig, axForce = plt.subplots(figsize=(8, 8), facecolor='w', ncols=1)

    fig.suptitle(f'Compression modulus')

    nSamples, colorSamples = getSamplesInfos(
        3, 2, 5,
        4, 2,
        'chocolate', 'mediumvioletred', 'steelblue',
        'lightcoral', 'crimson')

    raw_data, labels = getSamplesData(dataPath, nSamples)

    data = {
        labels[0]: ([], []),
        labels[1]: ([], []),
        labels[2]: ([], []),
        labels[3]: ([], []),
        labels[4]: ([], []),
    }

    for k, (x, y) in data.items():
        x.append(raw_data[f'{k} height to break'])
        y.append(raw_data[f'{k} force to break'])

    (fTitle, fLimits,
     hTitle, hLimits) = (
        f'Stress (Pa)', (0, 600),
        f'Strain', (0, 100))

    for k, c in zip(data, colorSamples):
        strain, stress = np.mean(data[k][0], axis=1)[0], np.mean(data[k][1], axis=1)[0]
        stressErr = np.std(data[k][0], axis=1)[0]

        plotCompression(
            ax=axForce, x=strain, y=stress, axisColor='k',
            axTitle='', yLabel=fTitle, yLim=fLimits, xLabel=hTitle, xLim=hLimits,
            curveColor=c, markerStyle='o', markerFColor=c, markerEColor='k',
            linearFitting=False, sampleName=f'{k}')

    # axForce.spines['left'].set_color(forceColor)
    # axForce.spines['right'].set_color(strainColor)

    plt.subplots_adjust(wspace=0.175, top=0.940, bottom=0.08, left=0.09, right=0.96)
    plt.show()

    dirSave = Path(*Path(filePath[0]).parts[:Path(filePath[0]).parts.index('data') + 1])
    fig.savefig(f'{dirSave}' + f'\\{fileName}' + '.png', facecolor='w', dpi=600)

    print(f'\n\n· Chart saved at\n{dirSave}.')


if __name__ == '__main__':
    folderPath = "C:/Users/petrus.kirsten/PycharmProjects/RheometerPlots/data"
    # folderPath = "C:/Users/Petrus Kirsten/Documents/GitHub/RheometerPlots/data"

    filePath = [
        # 0St/CL
        folderPath + "/171024/10_0St_CL/10_0St_CL-compression-1.xlsx",
        folderPath + "/171024/10_0St_CL/10_0St_CL-compression-2.xlsx",
        folderPath + "/171024/10_0St_CL/10_0St_CL-compression-3.xlsx",

        # 0St + kCar/CL
        folderPath + "/171024/10_0St_kC_CL/10_0St_kC_CL-compression-1.xlsx",
        folderPath + "/171024/10_0St_kC_CL/10_0St_kC_CL-compression-2.xlsx",

        # 0St + iCar/CL
        folderPath + "/171024/10_0St_iC_CL/10_0St_iC_CL-compression-1.xlsx",
        folderPath + "/171024/10_0St_iC_CL/10_0St_iC_CL-compression-2b.xlsx",
        folderPath + "/171024/10_0St_iC_CL/10_0St_iC_CL-compression-3.xlsx",
        folderPath + "/171024/10_0St_iC_CL/10_0St_iC_CL-compression-4.xlsx",
        folderPath + "/171024/10_0St_iC_CL/10_0St_iC_CL-compression-5.xlsx",

        # kC
        folderPath + "/231024/kC/kC-compression-1.xlsx",
        folderPath + "/231024/kC/kC-compression-2.xlsx",
        folderPath + "/231024/kC/kC-compression-3.xlsx",
        folderPath + "/231024/kC/kC-compression-4.xlsx",

        # kC/CL
        # folderPath + "/231024/kC_CL/kC_CL-compression-1.xlsx",   # 299
        # folderPath + "/231024/kC_CL/kC_CL-compression-1b.xlsx",  # 299
        folderPath + "/231024/kC_CL/kC_CL-compression-3.xlsx",   # 99
        folderPath + "/231024/kC_CL/kC_CL-compression-4.xlsx",   # 99
    ]

    main(filePath, '0St_Car_andCL-CompressionToBreakage')
