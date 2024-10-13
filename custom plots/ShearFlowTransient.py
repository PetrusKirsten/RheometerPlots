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


def exportFit(
        sample,
        data, err,
        rows, modHB=False):
    keys = ('eta0', 'tau0', 'tau_od', 'gammaDot_od', 'visc_K', 'visc_n', 'etaInf') if modHB \
        else ('K', 'n', 'sigmaZero')
    values = (data, err)

    dictData = {'Sample': sample}
    iParams = 0
    for key, value in zip(keys, range(len(values[0]))):
        dictData[f'{key}'] = values[0][iParams]
        dictData[f'{key} err'] = values[1][iParams]
        iParams += 1

    rows.append(dictData)

    return rows


def funcHB(sigma, k, n, sigmaZero):
    return sigmaZero + k * (sigma ** n)


def funcModHB(gamma_dot, eta_0, tau_0, tau_od, gamma_dot_od, K, n, eta_inf):
    # First part of the equation
    exp_term = np.exp(- (eta_0 * gamma_dot) / tau_0)
    part1 = 1 - exp_term

    # Second part (inside the curly brackets)
    term1 = (tau_0 - tau_od) / gamma_dot * np.exp(- gamma_dot / gamma_dot_od)
    term2 = tau_od / gamma_dot
    term3 = K * gamma_dot ** (n - 1)

    # Combine the second part
    part2 = term1 + term2 + term3

    # Final equation
    eta_ss = part1 * part2 + eta_inf

    return eta_ss


def funcTransient(t, tau_0, tau_e, alpha, gamma_dot):
    return tau_e + (tau_0 - tau_e) * np.exp(alpha * gamma_dot * t)


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


def getSamplesData(dataPath, n5st, nSt, nIc, nKc):
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
    samples = {'5% 0WSt kCar': [], '10% 0WSt': [], '10% 0WSt iCar': [], '10% 0WSt kCar': []}

    # Determine sample types for each path
    sample_labels = ['5% 0WSt kCar'] * n5st + ['10% 0WSt'] * nSt + ['10% 0WSt iCar'] * nIc + ['10% 0WSt kCar'] * nKc

    # Read data and categorize based on sample type
    for sample_type, path in zip(sample_labels, dataPath):
        df = pd.read_excel(path)
        segments = getSegments(df)
        samples[sample_type].append(segments)

    # Initialize dictionaries to hold the results
    dict_cteRate, dict_stepsRate = {}, {}

    # Populate dictionaries with consolidated sample data
    for sample_type in samples:
        dict_cteRate[f'{sample_type}_time'] = [s['time'][0] for s in samples[sample_type]]
        dict_cteRate[f'{sample_type}_rateCte'] = [s['shear_rate'][0] for s in samples[sample_type]]
        dict_cteRate[f'{sample_type}_stressCte'] = [s['shear_stress'][0] for s in samples[sample_type]]
        dict_cteRate[f'{sample_type}_viscosityCte'] = [s['viscosity'][0] for s in samples[sample_type]]

        dict_stepsRate[f'{sample_type}_rateSteps'] = [s['shear_rate'][1] for s in samples[sample_type]]
        dict_stepsRate[f'{sample_type}_stressSteps'] = [s['shear_stress'][1] for s in samples[sample_type]]
        dict_stepsRate[f'{sample_type}_viscositySteps'] = [s['viscosity'][1] for s in samples[sample_type]]

    return dict_cteRate, dict_stepsRate


def plotFlow(listRows, nSamples, sampleName,
             ax, x, y,
             axTitle, yLabel, yLim, xLabel, xLim,
             curveColor, markerStyle,
             individualData=False, fit='', logScale=False):
    def legendLabel():
        """Applies consistent styling to legends in plots."""
        legend = ax.legend(fancybox=False, frameon=True, framealpha=0.9, fontsize=9)
        legend.get_frame().set_facecolor('w')
        legend.get_frame().set_edgecolor('whitesmoke')

    def configPlot(idSample, xPlot, yPlot, yErr=0):
        ax.set_title(axTitle, size=9, color='crimson')
        ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)

        ax.set_xlabel(f'{xLabel}')
        ax.set_xscale('log' if logScale else 'linear')
        ax.set_xlim(xLim)

        ax.set_ylabel(f'{yLabel}')
        ax.set_yscale('log' if logScale else 'linear')
        ax.set_ylim(yLim)

        if fit != '':
            ax.plot(
                xPlot, yPlot, color=curveColor, linestyle=':', linewidth=1,
                zorder=2)
        else:
            ax.errorbar(
                xPlot, yPlot, yerr=yErr, color=curveColor, alpha=(0.9 - curve*0.2) if individualData else 1.0,
                fmt=markerStyle, markersize=7, mec='k', mew=0.5,
                capsize=3, lw=.5, linestyle='-',  # ecolor='k'
                label=f'{sampleName}', zorder=3)

        legendLabel()

    if individualData:
        for curve in range(nSamples):
            split_index = np.where(x[curve] <= 175)[0][-1]
            x_split, y_split = x[curve][:split_index], y[curve][:split_index]
            configPlot(curve, x_split, y_split)

            if fit == 'transient':
                params, covariance = curve_fit(funcTransient, x, y, p0=(10, 1, 0.1, 1), method='trf')  # method='dogbox', maxfev=5000)
                errors = np.sqrt(np.diag(covariance))
                print(params, errors)
                tau_0, tau_e, alpha, gamma_dot = params
                x_fit = np.linspace(0, 180, 180)
                y_fit = funcTransient(x_fit, tau_0, tau_e, alpha, gamma_dot)
                # listRows = exportFit(
                #     f'{sampleName}',
                #     params, errors,
                #     listRows)
                configPlot(curve, x_split, y_split)

    else:
        x_split, y_split = [None, None], [None, None]
        for curve in range(nSamples):
            split_index = np.where(x[curve] <= 175)[0][-1]
            x_split[curve], y_split[curve] = x[curve][:split_index], y[curve][:split_index]

        if sampleName != '5% 0WSt kCar':
            x_mean = np.mean(x_split, axis=0)
            y_err = np.std(y_split, axis=0)
            y_mean = np.mean(y_split, axis=0)
        else:
            split_index = np.where(x[0] <= 175)[0][-1]
            x_split, y_split = x[0][:split_index], y[0][:split_index]
            x_mean = x_split
            y_err = 0
            y_mean = y_split

        configPlot(0, x_mean, y_mean, yErr=y_err)

    return listRows


def main(dataPath):
    fonts('C:/Users/petrus.kirsten/AppData/Local/Microsoft/Windows/Fonts/')

    fileName = '10pct_0WSt_and_Car-Thixotropy'
    dirSave = Path(*Path(filePath[0]).parts[:Path(filePath[0]).parts.index('data') + 1])

    st5_nSamples, st10_nSamples, ic_nSamples, kc_nSamples = 1, 2, 2, 2
    constantShear, stepsShear = getSamplesData(dataPath, st5_nSamples, st10_nSamples, ic_nSamples, kc_nSamples)

    xTitle, xLimits = (f'Time (s)', (-4, 206))
    yTitle, yLimits = (f'Shear stress (Pa)', (0, 550))
    yTitleVisc, yLimitsVisc = f'Viscosity (mPa·s)', (yLimits[0]*3.33, yLimits[1]*3.33)
    st10_color, ic10_color, kc5_color, kc10_color = 'sandybrown', 'deepskyblue', 'lightpink', 'hotpink'

    plt.style.use('seaborn-v0_8-ticks')
    fig, axesStress = plt.subplots(figsize=(10, 7), facecolor='w', ncols=1, nrows=1)
    axesVisc = axesStress.twinx()
    axesVisc.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0)
    axesVisc.set_ylabel(f'{yTitleVisc}')
    axesVisc.set_ylim(yLimitsVisc)
    fig.suptitle(f'Constant shear rate flow')

    # Shear rate cte data
    fitModeStress, fitModeVisc = '', ''
    (x_10st, s_10st, v_10st,
     x_10ic, s_10ic, v_10ic,
     x_5kc, s_5kc, v_5kc,
     x_10kc, s_10kc, v_10kc) = (
        # 10% starch
        constantShear['10% 0WSt_time'],
        constantShear['10% 0WSt_stressCte'],
        constantShear['10% 0WSt_viscosityCte'],
        # 10% starch + iota
        constantShear['10% 0WSt iCar_time'],
        constantShear['10% 0WSt iCar_stressCte'],
        constantShear['10% 0WSt iCar_viscosityCte'],
        # 5% starch + kappa
        constantShear['5% 0WSt kCar_time'],
        constantShear['5% 0WSt kCar_stressCte'],
        constantShear['5% 0WSt kCar_viscosityCte'],
        # 10% starch + kappa
        constantShear['10% 0WSt kCar_time'],
        constantShear['10% 0WSt kCar_stressCte'],
        constantShear['10% 0WSt kCar_viscosityCte'])

    tableStress = []
    # Shear stress plot
    tableStress = plotFlow(
        listRows=tableStress, nSamples=st10_nSamples,
        ax=axesStress, x=x_10st, y=s_10st,
        axTitle='', yLabel=yTitle, yLim=yLimits, xLabel=xTitle, xLim=xLimits,
        curveColor=st10_color, markerStyle='o',
        sampleName=f'10% 0WSt', logScale=False, fit=fitModeStress)
    tableStress = plotFlow(
        listRows=tableStress, nSamples=ic_nSamples,
        ax=axesStress, x=x_10ic, y=s_10ic,
        axTitle='', yLabel=yTitle, yLim=yLimits, xLabel=xTitle, xLim=xLimits,
        curveColor=ic10_color, markerStyle='o',
        sampleName=f'10% 0WSt iCar', logScale=False, fit=fitModeStress)
    tableStress = plotFlow(
        listRows=tableStress, nSamples=st5_nSamples,
        ax=axesStress, x=x_5kc, y=s_5kc,
        axTitle='', yLabel=yTitle, yLim=yLimits, xLabel=xTitle, xLim=xLimits,
        curveColor=kc5_color, markerStyle='o',
        sampleName=f'5% 0WSt kCar', logScale=False, fit=fitModeStress)
    tableStress = plotFlow(
        listRows=tableStress, nSamples=kc_nSamples,
        ax=axesStress, x=x_10kc, y=s_10kc,
        axTitle='', yLabel=yTitle, yLim=yLimits, xLabel=xTitle, xLim=xLimits,
        curveColor=kc10_color, markerStyle='o',
        sampleName=f'10% 0WSt kCar', logScale=False, fit=fitModeStress)

    plt.subplots_adjust(hspace=0, wspace=0.200, top=0.940, bottom=0.095, left=0.090, right=0.900)
    # plt.tight_layout()
    plt.show()
    fig.savefig(f'{dirSave}' + f'\\{fileName}' + '.png', facecolor='w', dpi=600)

    fitParams = pd.DataFrame(tableStress)
    fitParams.to_excel(f'{dirSave}' + f'\\{fileName}' + '.xlsx', index=False)

    print(f'\n\n· Chart and tableStress with fitted parameters saved at\n{dirSave}.')


if __name__ == '__main__':
    # folderPath = "C:/Users/petrus.kirsten/PycharmProjects/RheometerPlots/data"
    folderPath = "C:/Users/Petrus Kirsten/Documents/GitHub/RheometerPlots/data"
    filePath = [
        #
        folderPath + "/091024/5_0WSt_kCar/5_0WSt_kCar-viscoRecoveryandFlow_1.xlsx",
        #
        folderPath + "/031024/10_0WSt/10_0WSt-viscRec_1.xlsx",
        folderPath + "/031024/10_0WSt/10_0WSt-viscRec_2.xlsx",
        #
        # folderPath + "10_0WSt_iCar/10_0WSt_iCar-viscoRecoveryandFlow_1.xlsx",
        folderPath + "/031024/10_0WSt_iCar/10_0WSt_iCar-viscoRecoveryandFlow_2.xlsx",
        # folderPath + "/031024/10_0WSt_iCar/10_0WSt_iCar-viscoRecoveryandFlow_3.xlsx",
        folderPath + "/031024/10_0WSt_iCar/10_0WSt_iCar-viscoRecoveryandFlow_4.xlsx",
        #
        folderPath + "/091024/10_0WSt_kCar/10_0WSt_kCar-viscoelasticRecovery-Flow_2a.xlsx",
        folderPath + "/091024/10_0WSt_kCar/10_0WSt_kCar-viscoelasticRecovery-Flow_3a.xlsx",
        # folderPath + "/091024/10_0WSt_kCar/10_0WSt_kCar-viscoelasticRecovery-Flow_4a.xlsx",
    ]

    main(dataPath=filePath)
