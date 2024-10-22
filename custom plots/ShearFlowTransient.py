import numpy as np
import pandas as pd
from pathlib import Path

from matplotlib.ticker import MultipleLocator
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


def exportFit(
        sample,
        data, err,
        rows, modHB=False):
    keys = ('Initial stress (tau_0) in Pa', 'Equilibrium stress (tau_e) in Pa', 'Characteristic time (lambda) in s')
    values = (data, err)

    dictData = {'Sample': sample}
    iParams = 0
    for key, value in zip(keys, range(len(values[0]))):
        dictData[f'{key}'] = values[0][iParams]
        dictData[f'{key} err'] = values[1][iParams]
        iParams += 1

    rows.append(dictData)

    return rows


def funcTransient(t, tau_0, tau_e, time_cte):
    return tau_e + (tau_0 - tau_e) * np.exp(- t / time_cte)


def getCteMean(values, tolerance=100):
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

    return mean, iStart, iEnd


def getSamplesData(dataPath, nSt, nKc, nIc, nStCL, nKcCL, nIcCL):
    """
    Reads multiple sample files and categorizes the data into 'cteRate' and 'stepsRate' dictionaries.
    """

    def getSegments(dataframe):
        """
        Extracts time, shear rate, shear stress, and viscosity segments from the dataframe.
        Returns tuples of constant and step segments.
        """
        time = dataframe['t in s'].to_numpy()
        shear_rate = dataframe['ɣ̇ in 1/s'].to_numpy()
        shear_stress = dataframe['τ in Pa'].to_numpy()
        viscosity = dataframe['η in mPas'].to_numpy()

        # Identifying segments in the data
        seg3, seg4, seg5 = (dataframe.index[dataframe['SegIndex'] == seg].to_list()[0] for seg in ['3|1', '4|1', '5|1'])

        # Slice segments
        segments = lambda arr: (arr[seg3:seg4], arr[seg4:seg5])  # Returns (constant segment, step segment)
        t_cte, t_steps = segments(time)

        return {
            'time': [t_cte - t_cte[0], t_steps - t_cte[0]],
            'shear_rate': segments(shear_rate),
            'shear_stress': segments(shear_stress),
            'viscosity': segments(viscosity)
        }

    # Store data for each sample type
    samples = {'0St': [], '0St + kCar': [], '0St + iCar': [],
               '0St/CL': [], '0St + kCar/CL': [], '0St + iCar/CL': []}
    # Determine sample types for each path
    sample_labels = (
            [list(samples.keys())[0]] * nSt + [list(samples.keys())[1]] * nKc + [list(samples.keys())[2]] * nIc +
            [list(samples.keys())[3]] * nStCL + [list(samples.keys())[4]] * nKcCL + [list(samples.keys())[5]] * nIcCL)
    # Read data and categorize based on sample type
    for sample_type, path in zip(sample_labels, dataPath):
        df = pd.read_excel(path)
        segments = getSegments(df)
        samples[sample_type].append(segments)

    # Initialize dictionaries to hold the results
    dict_cteRate = {}

    # Populate dictionaries with consolidated sample data
    for sample_type in samples:
        dict_cteRate[f'{sample_type}_time'] = [s['time'][0] for s in samples[sample_type]]

        dict_cteRate[f'{sample_type}_rateCte'] = [s['shear_rate'][0] for s in samples[sample_type]]
        dict_cteRate[f'{sample_type}_stressCte'] = [s['shear_stress'][0] for s in samples[sample_type]]
        dict_cteRate[f'{sample_type}_viscosityCte'] = [s['viscosity'][0] for s in samples[sample_type]]

    return dict_cteRate, list(samples.keys())


def plotFlow(listRows, nSamples, sampleName,
             ax, x, y,
             axTitle, yLabel, yLim, xLabel, xLim,
             curveColor, markerStyle,
             individualData=False, fit='', logScale=False):
    def legendLabel():
        """Applies consistent styling to legends in plots."""
        legend = ax.legend(
            fancybox=False, frameon=True,
            framealpha=0.9, fontsize=9, ncols=2,
            loc='lower right'
        )
        legend.get_frame().set_facecolor('w')
        legend.get_frame().set_edgecolor('whitesmoke')

    def configPlot(xPlot, yPlot, yErr=0, scatter=True):
        ax.set_title(axTitle, size=9, color='crimson')
        ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)

        ax.set_xlabel(f'{xLabel}')
        ax.set_xscale('log' if logScale else 'linear')
        ax.set_xlim(xLim)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))

        ax.set_ylabel(f'{yLabel}')
        ax.set_yscale('log' if logScale else 'linear')
        ax.set_ylim(yLim)
        ax.yaxis.set_major_locator(MultipleLocator(100))
        ax.yaxis.set_minor_locator(MultipleLocator(25))

        if scatter:
            ax.errorbar(
                xPlot, yPlot, yerr=yErr, color=curveColor, alpha=(0.9 - curve * 0.2) if individualData else .8,
                fmt=markerStyle, markersize=7, mec='k', mew=0.5,
                capsize=3, lw=.5, linestyle='',  # ecolor='k'
                label=f'{sampleName}', zorder=3)
        else:
            ax.plot(
                xPlot, yPlot, color=curveColor, linestyle='-.', linewidth=.75,
                zorder=2)

        legendLabel()

    if individualData:
        for curve in range(nSamples):
            split_index = np.where(x[curve] <= 175)[0][-1]
            x_split, y_split = x[curve][:split_index], y[curve][:split_index]
            configPlot(x_split, y_split)

            if fit == 'transient':
                params, covariance = curve_fit(funcTransient, x, y, p0=(10, 1, 0.1, 1),
                                               method='trf')  # method='dogbox', maxfev=5000)
                errors = np.sqrt(np.diag(covariance))
                print(params, errors)
                tau_0, tau_e, t_cte = params
                x_fit = np.linspace(0, 180, 180)
                y_fit = funcTransient(x_fit, tau_0, tau_e, t_cte)
                # listRows = exportFit(
                #     f'{sampleName}',
                #     params, errors,
                #     listRows)
                configPlot(x_split, y_split)

    else:
        x_split, y_split = [], []
        for curve in range(nSamples):
            split_index = np.where(x[curve] <= 175)[0][-1]
            x_split.append(x[curve][:split_index]), y_split.append(y[curve][:split_index])

        x_mean = np.mean(x_split, axis=0)
        y_err = np.std(y_split, axis=0)
        y_mean = np.mean(y_split, axis=0)

        if fit == 'transient':
            params, covariance = curve_fit(funcTransient, x_mean, y_mean, p0=(y_mean[0], y_mean[-1], 100))
            # method='trf')  # method='dogbox', maxfev=5000)
            errors = np.sqrt(np.diag(covariance))
            print(params, errors)
            tau_0, tau_e, t_cte = params
            x_fit = np.linspace(0, 300, 300)
            y_fit = funcTransient(x_fit, tau_0, tau_e, t_cte)
            listRows = exportFit(
                f'{sampleName}',
                params, errors,
                listRows)
            configPlot(x_fit, y_fit, scatter=False)

        configPlot(x_mean, y_mean, yErr=y_err)

    return listRows


def main(dataPath):
    fonts('C:/Users/petrus.kirsten/AppData/Local/Microsoft/Windows/Fonts/')

    fileName = 'St_Car_CL-Thixotropy'
    dirSave = Path(*Path(filePath[0]).parts[:Path(filePath[0]).parts.index('data') + 1])

    fig, axesStress = plt.subplots(figsize=(12, 7), facecolor='w', ncols=1, nrows=1)
    fig.suptitle(f'Constant shear rate flow')
    plt.style.use('seaborn-v0_8-ticks')
    xTitle, xLimits = (f'Time (s)', (0, 240))
    yTitle, yLimits = (f'Shear stress (Pa)', (0, 550))
    yTitleVisc, yLimitsVisc = f'Viscosity (mPa·s)', (yLimits[0] * 3.33, yLimits[1] * 3.33)

    axesVisc = axesStress.twinx()
    axesVisc.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0)
    axesVisc.set_ylabel(f'{yTitleVisc}')
    axesVisc.set_ylim(yLimitsVisc)
    axesVisc.yaxis.set_major_locator(MultipleLocator(200))
    axesVisc.yaxis.set_minor_locator(MultipleLocator(50))

    (st_n, kc_n, ic_n,
     stCL_n, kcCL_n, icCL_n) = (2, 2, 2,
                                3, 1, 3)
    nSamples = (st_n, kc_n, ic_n,
                stCL_n, kcCL_n, icCL_n)

    (st_color, kc_color, ic_color,
     stCL_color, kcCL_color, icCL_color) = ('sandybrown', 'hotpink', 'deepskyblue',
                                            'chocolate', 'mediumvioletred', 'steelblue')
    colors = (st_color, kc_color, ic_color,
              stCL_color, kcCL_color, icCL_color)

    constantShear, sampleLabels = getSamplesData(dataPath, *nSamples)

    fitModeStress, fitModeVisc = 'transient', ''

    (x_st, s_st,
     x_kc, s_kc,
     x_ic, s_ic,
     x_stCL, s_stCL,
     x_kcCL, s_kcCL,
     x_icCL, s_icCL) = (
        # 10% starch
        constantShear[f'{sampleLabels[0]}_time'],
        constantShear[f'{sampleLabels[0]}_stressCte'],
        # 10% starch + kappa
        constantShear[f'{sampleLabels[1]}_time'],
        constantShear[f'{sampleLabels[1]}_stressCte'],
        # 10% starch + iota
        constantShear[f'{sampleLabels[2]}_time'],
        constantShear[f'{sampleLabels[2]}_stressCte'],
        # 10% starch CL
        constantShear[f'{sampleLabels[3]}_time'],
        constantShear[f'{sampleLabels[3]}_stressCte'],
        # 10% starch + kappa CL
        constantShear[f'{sampleLabels[4]}_time'],
        constantShear[f'{sampleLabels[4]}_stressCte'],
        # 10% starch + iota CL
        constantShear[f'{sampleLabels[5]}_time'],
        constantShear[f'{sampleLabels[5]}_stressCte'],)

    data = [(x_st, s_st),
            (x_kc, s_kc),
            (x_ic, s_ic),
            (x_stCL, s_stCL),
            (x_kcCL, s_kcCL),
            (x_icCL, s_icCL)]

    tableStress = []

    for i in range(len(data)):
        tableStress = plotFlow(
            listRows=tableStress, nSamples=nSamples[i],
            ax=axesStress, x=data[i][0], y=data[i][1],
            axTitle='', yLabel=yTitle, yLim=yLimits, xLabel=xTitle, xLim=xLimits,
            curveColor=colors[i], markerStyle='o',
            sampleName=f'{sampleLabels[i]}', fit=fitModeStress)

    plt.subplots_adjust(
        hspace=0,
        wspace=0.200,
        top=0.940,
        bottom=0.095,
        left=0.090,
        right=0.900)
    # plt.tight_layout()
    plt.show()
    fig.savefig(f'{dirSave}' + f'\\{fileName}' + '.png', facecolor='w', dpi=600)

    fitParams = pd.DataFrame(tableStress)
    fitParams.to_excel(f'{dirSave}' + f'\\{fileName}' + '.xlsx', index=False)

    print(f'\n\n· Chart and tableStress with fitted parameters saved at\n{dirSave}.')


if __name__ == '__main__':
    folderPath = "C:/Users/petrus.kirsten/PycharmProjects/RheometerPlots/data"
    # folderPath = "C:/Users/Petrus Kirsten/Documents/GitHub/RheometerPlots/data"
    filePath = [
        #
        folderPath + "/031024/10_0WSt/10_0WSt-viscRec_1.xlsx",
        folderPath + "/031024/10_0WSt/10_0WSt-viscRec_2.xlsx",
        #
        folderPath + "/091024/10_0WSt_kCar/10_0WSt_kCar-viscoelasticRecovery-Flow_2a.xlsx",
        folderPath + "/091024/10_0WSt_kCar/10_0WSt_kCar-viscoelasticRecovery-Flow_3a.xlsx",
        #
        # folderPath + "10_0WSt_iCar/10_0WSt_iCar-viscoRecoveryandFlow_1.xlsx",
        folderPath + "/031024/10_0WSt_iCar/10_0WSt_iCar-viscoRecoveryandFlow_2.xlsx",
        # folderPath + "/031024/10_0WSt_iCar/10_0WSt_iCar-viscoRecoveryandFlow_3.xlsx",
        folderPath + "/031024/10_0WSt_iCar/10_0WSt_iCar-viscoRecoveryandFlow_4.xlsx",
        # folderPath + "/091024/10_0WSt_kCar/10_0WSt_kCar-viscoelasticRecovery-Flow_4a.xlsx",
        #
        folderPath + "/171024/10_0St_CL/10_0St_CL-recovery-1.xlsx",
        folderPath + "/171024/10_0St_CL/10_0St_CL-recovery-2.xlsx",
        folderPath + "/171024/10_0St_CL/10_0St_CL-recovery-3.xlsx",
        #
        folderPath + "/171024/10_0St_kC_CL/10_0St_kC_CL-recovery-1.xlsx",
        # folderPath + "/171024/10_0St_kC_CL/10_0St_CL-recovery-2.xlsx",
        #
        folderPath + "/171024/10_0St_iC_CL/10_0St_iC_CL-recovery-1.xlsx",
        folderPath + "/171024/10_0St_iC_CL/10_0St_iC_CL-recovery-2.xlsx",
        folderPath + "/171024/10_0St_iC_CL/10_0St_iC_CL-recovery-3.xlsx",
    ]

    main(dataPath=filePath)
