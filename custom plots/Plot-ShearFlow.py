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
        K, n, sigmaZero, err,
        rows):
    data = {
        'Sample': sample,
        'K': K, 'K err': err[0],
        'n': n, 'n err': err[1],
        'sigmaZero': sigmaZero, 'sigmaZero err': err[2]}

    rows.append(data)

    return rows


# TODO: fazer função para fitar modelo de HB
def powerLaw(sigma, k, n, sigmaZero):
    return sigmaZero + k * (sigma ** n)


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


def columnsRead(dataframe):
    time, shearRate, shearStress, viscosity = (
        dataframe['t in s'].to_numpy(),
        dataframe['ɣ̇ in 1/s'].to_numpy(),
        dataframe['τ in Pa'].to_numpy(),
        dataframe['η in mPas'].to_numpy())

    seg3, seg4, seg5 = (dataframe.index[dataframe['SegIndex'] == '3|1'].to_list()[0],
                        dataframe.index[dataframe['SegIndex'] == '4|1'].to_list()[0],
                        dataframe.index[dataframe['SegIndex'] == '5|1'].to_list()[0])

    tCte, tSteps = time[seg3:seg4], time[seg4:seg5]
    shearRate_cte, shearRate_steps = shearRate[seg3:seg4], shearRate[seg4:seg5]
    shearStress_cte, shearStress_steps = shearStress[seg3:seg4], shearStress[seg4:seg5]
    viscosity_cte, viscosity_steps = viscosity[seg3:seg4], viscosity[seg4:seg5]

    return ([tCte - tCte[0], tSteps - tCte[0]],
            [shearRate_cte, shearRate_steps],
            [shearStress_cte, shearStress_steps],
            [viscosity_cte, viscosity_steps])


def getSamplesData(dataPath, nSt, nIc):
    dict_cteRate, dict_stepsRate = {}, {}  # Dicionário para armazenar resultados por caminho de arquivo
    st_time, st_rateCte, st_rateSteps, st_stressCte, st_stressSteps, st_viscosityCte, st_viscositySteps = [], [], [], [], [], [], []
    ic_time, ic_rateCte, ic_rateSteps, ic_stressCte, ic_stressSteps, ic_viscosityCte, ic_viscositySteps = [], [], [], [], [], [], []

    for sample, path in enumerate(dataPath):
        df = pd.read_excel(path)

        if sample < nSt:
            st_time_i, st_rate_i, st_stress_i, st_viscosity_i = columnsRead(df)

            st_time.append(st_time_i[0])

            st_rateCte.append(st_rate_i[0])
            st_rateSteps.append(st_rate_i[1])

            st_stressCte.append(st_stress_i[0])
            st_stressSteps.append(st_stress_i[1])

            st_viscosityCte.append(st_viscosity_i[0])
            st_viscositySteps.append(st_viscosity_i[1])

        else:
            ic_time_i, ic_rate_i, ic_stress_i, ic_viscosity_i = columnsRead(df)

            ic_time.append(ic_time_i[0])

            ic_rateCte.append(ic_rate_i[0])
            ic_rateSteps.append(ic_rate_i[1])

            ic_stressCte.append(ic_stress_i[0])
            ic_stressSteps.append(ic_stress_i[1])

            ic_viscosityCte.append(ic_viscosity_i[0])
            ic_viscositySteps.append(ic_viscosity_i[1])

    # Cte shear rate data
    #   Pure starch
    (dict_cteRate[f'st_time'],
     dict_cteRate[f'st_rateCte'],
     dict_cteRate[f'st_stressCte'],
     dict_cteRate[f'st_viscosityCte']) = st_time, st_rateCte, st_stressCte, st_viscosityCte

    (dict_stepsRate[f'st_rateSteps'],
     dict_stepsRate[f'st_stressSteps'],
     dict_stepsRate[f'st_viscositySteps']) = st_rateSteps, st_stressSteps, st_viscositySteps

    #   Starch + iCar
    (dict_cteRate[f'ic_time'],
     dict_cteRate[f'ic_rateCte'],
     dict_cteRate[f'ic_stressCte'],
     dict_cteRate[f'ic_viscosityCte']) = ic_time, ic_rateCte, ic_stressCte, ic_viscosityCte

    (dict_stepsRate[f'ic_rateSteps'],
     dict_stepsRate[f'ic_stressSteps'],
     dict_stepsRate[f'ic_viscositySteps']) = ic_rateSteps, ic_stressSteps, ic_viscositySteps

    return dict_cteRate, dict_stepsRate


def plotFlowTime(
        ax, x, y,
        axTitle, yLabel, yLim,
        curveColor, markerStyle,
        sampleName,
        logScale=False,
        fit=False):
    ax.set_title(axTitle, size=9, color='crimson')
    ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)
    ax.set_xlabel('Time (s)')
    ax.set_xscale('log' if logScale else 'linear')
    ax.set_xlim([-65, +1650])

    ax.set_ylabel(f'{yLabel}')
    ax.set_yscale('log' if logScale else 'linear')
    ax.set_ylim(yLim)

    if fit:
        ax.plot(
            x, y, color=curveColor, linestyle=':', linewidth=1,
            zorder=2)
    else:
        ax.errorbar(
            x[::3], y[::3], yerr=0, color=curveColor, alpha=0.8,
            fmt=markerStyle, markersize=7, mec='k', mew=0.5,
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

    fileName = '10pct_0WSt_and_iCar-Thixotropy_031024'
    dirSave = f'{Path(filePath[0]).parent}' + f'\\{fileName}' + '.png'

    plt.style.use('seaborn-v0_8-ticks')
    fig, axStress = plt.subplots(figsize=(10, 7), facecolor='w', ncols=1)
    axVisc = axStress.twinx()
    fig.suptitle(f'Shear flow')

    constantShear, _ = getSamplesData(dataPath, 2, 3)

    x_st, s_st, v_st, x_ic, s_ic, v_ic = (
        constantShear['st_time'],
        constantShear['st_stressCte'],
        constantShear['st_viscosityCte'],
        #
        constantShear['ic_time'],
        constantShear['ic_stressCte'],
        constantShear['ic_viscosityCte'])

    tableRows = []
    for curve in range(2):
        st_params, st_covariance = curve_fit(powerLaw, x_st[curve], s_st[curve], p0=(2, 1, 0))
        st_errors = np.sqrt(np.diag(st_covariance))
        st_K, st_n, st_sigmaZero = st_params
        x_fit = np.linspace(0, 700, 700)
        y_fit = powerLaw(x_fit, st_K, st_n, st_sigmaZero)

        plotFlowTime(
            axStress, x_fit.tolist(), y_fit.tolist(),
            axTitle='', yLabel='', yLim=(100, 600),
            curveColor='silver', markerStyle=None,
            sampleName=f'',
            fit=True)
        plotFlowTime(
            axStress, x_st[curve].tolist(), s_st[curve],
            axTitle='', yLabel='Shear stress (Pa)', yLim=(100, 600),
            curveColor='silver', markerStyle='o',
            sampleName=f'10%_0WSt_{curve + 1}')
        plotFlowTime(
            axVisc, (x_st[curve] + 800).tolist(), v_st[curve],
            axTitle='', yLabel='Viscosity (mPa·s)', yLim=(400, 2000),
            curveColor='silver', markerStyle='s',
            sampleName=f'')

        tableRows = exportFit(
            f'10%_0WSt_{curve + 1}',
            st_K, st_n, st_sigmaZero, st_errors,
            tableRows)
        print(
            f'\n10_0WSt thixotropy fit parameters:\n'
            f'K = {st_K:.2f} ± {st_errors[0]:.2f},\n'
            f'n = {st_n:.2f} ± {st_errors[1]:.2f},\n'
            f'sigma_0 = {st_sigmaZero:.1f} ± {st_errors[2]:.2f}\n')

    for curve in range(3):
        ic_params, ic_covariance = curve_fit(powerLaw, x_ic[curve], s_ic[curve])
        ic_errors = np.sqrt(np.diag(ic_covariance))
        ic_K, ic_n, ic_sigmaZero = ic_params
        x_fit = np.linspace(0, 700, 700)
        y_fit = powerLaw(x_fit, ic_K, ic_n, ic_sigmaZero)

        plotFlowTime(
            axStress, x_fit.tolist(), y_fit.tolist(),
            axTitle='', yLabel='', yLim=(100, 600),
            curveColor='deepskyblue', markerStyle=None,
            sampleName=f'',
            fit=True)
        plotFlowTime(
            axStress, x_ic[curve].tolist(), s_ic[curve],
            axTitle='', yLabel='Shear stress (Pa)', yLim=(100, 600),
            curveColor='deepskyblue', markerStyle='o',
            sampleName=f'10%_0WSt_iCar_{curve + 1}')
        plotFlowTime(
            axVisc, (x_ic[curve] + 800).tolist(), v_ic[curve],
            axTitle='', yLabel='Viscosity (mPa·s)', yLim=(400, 2000),
            curveColor='deepskyblue', markerStyle='s',
            sampleName=f'')

        tableRows = exportFit(
            f'10%_0WSt_iCar_{curve + 1}',
            ic_K, ic_n, ic_sigmaZero, ic_errors,
            tableRows)
        print(
            f'\n10_0WSt_iCar_{curve + 1} thixotropy fit parameters:\n'
            f'K = {ic_K:.2f} ± {ic_errors[0]:.2f},\n'
            f'n = {ic_n:.2f} ± {ic_errors[1]:.2f},\n'
            f'sigma_0 = {ic_sigmaZero:.1f} ± {ic_errors[2]:.2f}\n')

    fitParams = pd.DataFrame(tableRows)
    fitParams.to_excel(f'{fileName}_fit.xlsx', index=False)

    # plt.subplots_adjust(wspace=0.175, top=0.890, bottom=0.14, left=0.05, right=0.95)
    plt.tight_layout()
    # fig.savefig(dirSave, facecolor='w', dpi=600)
    plt.show()


if __name__ == '__main__':
    # filePath = ('C:/Users/Petrus Kirsten/Documents/GitHub/RheometerPlots/data/031024/10pct_0WSt/10pct_0WSt'
    #             '-RecoveryAndFlow_2.xlsx')  # personal PC

    folderPath = "C:/Users/petrus.kirsten/PycharmProjects/RheometerPlots/data"
    filePath = [
        folderPath + "/031024/10_0WSt/10_0WSt-viscRec_1.xlsx",
        folderPath + "/031024/10_0WSt/10_0WSt-viscRec_2.xlsx",
        #
        # folderPath + "10_0WSt_iCar/10_0WSt_iCar-viscoRecoveryandFlow_1.xlsx",
        folderPath + "/031024/10_0WSt_iCar/10_0WSt_iCar-viscoRecoveryandFlow_2.xlsx",
        folderPath + "/031024/10_0WSt_iCar/10_0WSt_iCar-viscoRecoveryandFlow_3.xlsx",
        folderPath + "/031024/10_0WSt_iCar/10_0WSt_iCar-viscoRecoveryandFlow_4.xlsx"]

    main(dataPath=filePath)
