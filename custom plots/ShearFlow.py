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


def getSamplesData(dataPath, nSt, nIc):
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
    samples = {'st': [], 'ic': [], 'kc': []}

    # Determine sample types for each path
    sample_labels = ['st'] * nSt + ['ic'] * nIc + ['kc'] * (len(dataPath) - nSt - nIc)

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


def plotFlow(listRows, nSamples,
             ax, x, y,
             axTitle, yLabel, yLim, xLabel, xLim,
             curveColor, markerStyle,
             sampleName, fit='',
             logScale=False):
    def legendLabel():
        """Applies consistent styling to legends in plots."""
        legend = ax.legend(fancybox=False, frameon=True, framealpha=0.9, fontsize=9)
        legend.get_frame().set_facecolor('w')
        legend.get_frame().set_edgecolor('whitesmoke')

    def configPlot(idSample):
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
                x_fit, y_fit, color=curveColor, linestyle=':', linewidth=1,
                zorder=2)

        ax.errorbar(
            x[curve], y[curve], yerr=0, color=curveColor, alpha=(0.9 - curve * 0.2),
            fmt=markerStyle, markersize=7, mec='k', mew=0.5,
            capsize=3, lw=1, linestyle='',  # ecolor='k'
            label=f'{sampleName}_{idSample + 1}', zorder=3)

        legendLabel()

    for curve in range(nSamples):
        # split_index = np.where(x[curve] <= 180)[0][-1]
        # x[curve], y[curve] = x[curve][:split_index], y[curve][:split_index]

        if fit == 'HB':
            params, covariance = curve_fit(funcHB, x[curve], y[curve], p0=(2, 1, 0))
            errors = np.sqrt(np.diag(covariance))
            K, n, sigmaZero = params
            x_fit = np.linspace(.1, 1000, 1000)
            y_fit = funcHB(x_fit, K, n, sigmaZero)
            listRows = exportFit(
                f'{sampleName}',
                params, errors,
                listRows)
            configPlot(idSample=curve)

        if fit == 'modHB':
            params, covariance = curve_fit(funcModHB, x[curve], y[curve],
                                           p0=(100, 10, 5, 1, 1, 0.5, 0.1), maxfev=5000, method='dogbox')
            errors = np.sqrt(np.diag(covariance))
            eta0, tau0, tau_od, gammaDot_od, visc_K, visc_n, etaInf = params
            x_fit = np.linspace(.1, 1000, 1000)
            y_fit = funcModHB(x_fit, eta0, tau0, tau_od, gammaDot_od, visc_K, visc_n, etaInf)
            listRows = exportFit(
                f'{sampleName}',
                params, errors,
                listRows, modHB=True)
            configPlot(idSample=curve)

        if fit == 'transient':
            params, covariance = curve_fit(funcTransient, x[curve], y[curve], p0=(10, 1, 0.1, 1), method='trf')  # method='dogbox', maxfev=5000)
            errors = np.sqrt(np.diag(covariance))
            print(params, errors)
            tau_0, tau_e, alpha, gamma_dot = params
            x_fit = np.linspace(0, 180, 180)
            y_fit = funcTransient(x_fit, tau_0, tau_e, alpha, gamma_dot)
            # listRows = exportFit(
            #     f'{sampleName}',
            #     params, errors,
            #     listRows)
            configPlot(idSample=curve)

        configPlot(idSample=curve)

    return listRows


def main(dataPath, thixo):
    fonts('C:/Users/petrus.kirsten/AppData/Local/Microsoft/Windows/Fonts/')

    fileName = '10pct_0WSt_and_Car-Thixotropy' if thixo else '10pct_0WSt_and_Car-Flow'
    dirSave = Path(*Path(filePath[0]).parts[:Path(filePath[0]).parts.index('data') + 1])

    plt.style.use('seaborn-v0_8-ticks')
    fig, axStress = plt.subplots(figsize=(10, 7), facecolor='w', ncols=1)
    axVisc = None if thixo else axStress.twinx()
    fig.suptitle(f'Constante shear rate flow' if thixo else f'Steps shear rate flow')

    xTitle, xLimits = (f'Time (s)', (-5, 200)) if thixo else (f'Shear rate ($s^{-1}$)', (15, 500))
    yTitle, yLimits = (f'Shear stress (Pa)', (100, 600)) if thixo else (f'Shear stress (Pa)', (3, 550))
    yTitleVisc, yLimitsVisc = f'Viscosity (mPa·s)', (450, 100000)
    st5_color, st10_color, ic_color, kc_color = 'silver', 'sandybrown', 'hotpink', 'mediumturquoise'

    st_nSamples, ic_nSamples, kc_nSamples = 2, 3, 2
    constantShear, stepsShear = getSamplesData(dataPath, st_nSamples, ic_nSamples)

    # Shear rate cte data
    if thixo:
        fitModeStress, fitModeVisc = '', ''
        (x_st, s_st, v_st,
         x_ic, s_ic, v_ic,
         x_kc, s_kc, v_kc) = (
            # 10% starch
            constantShear['st_time'],
            constantShear['st_stressCte'],
            constantShear['st_viscosityCte'],
            # 10% starch + iota
            constantShear['ic_time'],
            constantShear['ic_stressCte'],
            constantShear['ic_viscosityCte'],
            # 10% starch + kappa
            constantShear['kc_time'],
            constantShear['kc_stressCte'],
            constantShear['kc_viscosityCte'])
    # Shear rate steps data
    else:
        fitModeStress, fitModeVisc = 'HB', 'modHB'
        (x_st, s_st, v_st,
         x_ic, s_ic, v_ic,
         x_kc, s_kc, v_kc) = (
            # 10% starch
            stepsShear['st_rateSteps'],
            stepsShear['st_stressSteps'],
            stepsShear['st_viscositySteps'],
            # 10% starch + iota
            stepsShear['ic_rateSteps'],
            stepsShear['ic_stressSteps'],
            stepsShear['ic_viscositySteps'],
            # 10% starch + kappa
            stepsShear['kc_rateSteps'],
            stepsShear['kc_stressSteps'],
            stepsShear['kc_viscositySteps'])

    tableStress = []
    # Shear stress plot
    tableStress = plotFlow(
        listRows=tableStress, nSamples=st_nSamples,
        ax=axStress, x=x_st, y=s_st,
        axTitle='', yLabel=yTitle, yLim=yLimits, xLabel=xTitle, xLim=xLimits,
        curveColor=st10_color, markerStyle='o',
        sampleName=f's10_0WSt', logScale=False if thixo else True, fit=fitModeStress)

    tableStress = plotFlow(
        listRows=tableStress, nSamples=ic_nSamples,
        ax=axStress, x=x_ic, y=s_ic,
        axTitle='', yLabel=yTitle, yLim=yLimits, xLabel=xTitle, xLim=xLimits,
        curveColor=ic_color, markerStyle='o',
        sampleName=f's10_0WSt_iCar', logScale=False if thixo else True, fit=fitModeStress)

    tableStress = plotFlow(
        listRows=tableStress, nSamples=kc_nSamples,
        ax=axStress, x=x_kc, y=s_kc,
        axTitle='', yLabel=yTitle, yLim=yLimits, xLabel=xTitle, xLim=xLimits,
        curveColor=kc_color, markerStyle='o',
        sampleName=f's10_0WSt_kCar', logScale=False if thixo else True, fit=fitModeStress)

    # Viscosity plot
    # tableStress = plotFlow(
    #     listRows=tableStress, nSamples=st_nSamples,
    #     ax=axVisc, x=x_st, y=v_st,
    #     axTitle='', yLabel=yTitleVisc, yLim=yLimitsVisc, xLabel=xTitle, xLim=xLimits,
    #     curveColor=st10_color, markerStyle='s',
    #     sampleName=f'v10_0WSt', logScale=True, fit=fitModeVisc)
    #
    # tableStress = plotFlow(
    #     listRows=tableStress, nSamples=ic_nSamples,
    #     ax=axVisc, x=x_ic, y=v_ic,
    #     axTitle='', yLabel=yTitleVisc, yLim=yLimitsVisc, xLabel=xTitle, xLim=xLimits,
    #     curveColor=ic_color, markerStyle='s',
    #     sampleName=f'v10_0WSt_iCar', logScale=True, fit=fitModeVisc)
    #
    # tableStress = plotFlow(
    #     listRows=tableStress, nSamples=kc_nSamples,
    #     ax=axVisc, x=x_kc, y=v_kc,
    #     axTitle='', yLabel=yTitleVisc, yLim=yLimitsVisc, xLabel=xTitle, xLim=xLimits,
    #     curveColor=kc_color, markerStyle='s',
    #     sampleName=f'v10_0WSt_kCar', logScale=True, fit=fitModeVisc)

    # plt.subplots_adjust(wspace=0.175, top=0.890, bottom=0.14, left=0.05, right=0.95)
    plt.tight_layout()
    plt.show()
    fig.savefig(f'{dirSave}' + f'\\{fileName}' + '.png', facecolor='w', dpi=600)

    fitParams = pd.DataFrame(tableStress)
    fitParams.to_excel(f'{dirSave}' + f'\\{fileName}' + '.xlsx', index=False)

    print(f'\n\n· Chart and tableStress with fitted parameters saved at\n{dirSave}.')


if __name__ == '__main__':
    folderPath = "C:/Users/petrus.kirsten/PycharmProjects/RheometerPlots/data"
    # folderPath = "C:/Users/Petrus Kirsten/Documents/GitHub/RheometerPlots/data"
    filePath = [
        folderPath + "/031024/10_0WSt/10_0WSt-viscRec_1.xlsx",
        folderPath + "/031024/10_0WSt/10_0WSt-viscRec_2.xlsx",
        #
        # folderPath + "10_0WSt_iCar/10_0WSt_iCar-viscoRecoveryandFlow_1.xlsx",
        folderPath + "/031024/10_0WSt_iCar/10_0WSt_iCar-viscoRecoveryandFlow_2.xlsx",
        folderPath + "/031024/10_0WSt_iCar/10_0WSt_iCar-viscoRecoveryandFlow_3.xlsx",
        folderPath + "/031024/10_0WSt_iCar/10_0WSt_iCar-viscoRecoveryandFlow_4.xlsx",
        #
        folderPath + "/091024/10_0WSt_kCar/10_0WSt_kCar-viscoelasticRecovery-Flow_2a.xlsx",
        folderPath + "/091024/10_0WSt_kCar/10_0WSt_kCar-viscoelasticRecovery-Flow_3a.xlsx",
        # folderPath + "/091024/10_0WSt_kCar/10_0WSt_kCar-viscoelasticRecovery-Flow_4a.xlsx",
    ]

    main(dataPath=filePath, thixo=True)
