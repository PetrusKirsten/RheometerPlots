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
    kc_time, kc_rateCte, kc_rateSteps, kc_stressCte, kc_stressSteps, kc_viscosityCte, kc_viscositySteps = [], [], [], [], [], [], []

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

        elif sample < nSt + nIc:
            ic_time_i, ic_rate_i, ic_stress_i, ic_viscosity_i = columnsRead(df)

            ic_time.append(ic_time_i[0])

            ic_rateCte.append(ic_rate_i[0])
            ic_rateSteps.append(ic_rate_i[1])

            ic_stressCte.append(ic_stress_i[0])
            ic_stressSteps.append(ic_stress_i[1])

            ic_viscosityCte.append(ic_viscosity_i[0])
            ic_viscositySteps.append(ic_viscosity_i[1])

        else:
            kc_time_i, kc_rate_i, kc_stress_i, kc_viscosity_i = columnsRead(df)

            kc_time.append(kc_time_i[0])

            kc_rateCte.append(kc_rate_i[0])
            kc_rateSteps.append(kc_rate_i[1])

            kc_stressCte.append(kc_stress_i[0])
            kc_stressSteps.append(kc_stress_i[1])

            kc_viscosityCte.append(kc_viscosity_i[0])
            kc_viscositySteps.append(kc_viscosity_i[1])

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

    #   Starch + kCar
    (dict_cteRate[f'kc_time'],
     dict_cteRate[f'kc_rateCte'],
     dict_cteRate[f'kc_stressCte'],
     dict_cteRate[f'kc_viscosityCte']) = kc_time, kc_rateCte, kc_stressCte, kc_viscosityCte

    (dict_stepsRate[f'kc_rateSteps'],
     dict_stepsRate[f'kc_stressSteps'],
     dict_stepsRate[f'kc_viscositySteps']) = kc_rateSteps, kc_stressSteps, kc_viscositySteps

    return dict_cteRate, dict_stepsRate


def plotFlowTime(listRows, nSamples,
                 ax, x, y,
                 axTitle, yLabel, yLim,
                 curveColor, markerStyle,
                 sampleName,
                 logScale=False):
    def legendLabel():
        """Applies consistent styling to legends in plots."""
        legend = ax.legend(fancybox=False, frameon=True, framealpha=0.9, fontsize=9)
        legend.get_frame().set_facecolor('w')
        legend.get_frame().set_edgecolor('whitesmoke')

    def configPlot(fit, idSample):
        ax.set_title(axTitle, size=9, color='crimson')
        ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)
        ax.set_xlabel('Time (s)')
        ax.set_xscale('log' if logScale else 'linear')
        ax.set_xlim([-20, +600])

        ax.set_ylabel(f'{yLabel}')
        ax.set_yscale('log' if logScale else 'linear')
        ax.set_ylim(yLim)

        if fit:
            ax.plot(
                x_fit, y_fit, color=curveColor, linestyle=':', linewidth=1,
                zorder=2)
        else:
            ax.errorbar(
                x[curve][::3], y[curve][::3], yerr=0, color=curveColor, alpha=(0.9 - curve*0.2),
                fmt=markerStyle, markersize=7, mec='k', mew=0.5,
                capsize=3, lw=1, linestyle='',  # ecolor='k'
                label=f'{sampleName}_{idSample + 1}', zorder=3)

        legendLabel()

    for curve in range(nSamples):
        params, covariance = curve_fit(powerLaw, x[curve], y[curve], p0=(2, 1, 0))
        errors = np.sqrt(np.diag(covariance))
        K, n, sigmaZero = params
        x_fit = np.linspace(0, 700, 700)
        y_fit = powerLaw(x_fit, K, n, sigmaZero)

        configPlot(fit=False, idSample=curve)
        configPlot(fit=True, idSample=curve)

        listRows = exportFit(
            f'{sampleName}',
            K, n, sigmaZero, errors,
            listRows)

        print(
            f'\n· {sampleName} thixotropy fit parameters:\n'
            f'K = {K:.2f} ± {errors[0]:.2f},\n'
            f'n = {n:.2f} ± {errors[1]:.2f},\n'
            f'sigma_0 = {sigmaZero:.1f} ± {errors[2]:.2f}.\n')

    return listRows


def main(dataPath):
    fonts('C:/Users/petrus.kirsten/AppData/Local/Microsoft/Windows/Fonts/')
    # samplesQuantities = list(samplesValues.keys())

    fileName = '10pct_0WSt_and_Car-Thixotropy'
    dirSave = Path(*Path(filePath[0]).parts[:Path(filePath[0]).parts.index('data') + 1])

    plt.style.use('seaborn-v0_8-ticks')
    fig, axStress = plt.subplots(figsize=(10, 7), facecolor='w', ncols=1)
    # axVisc = axStress.twinx()
    fig.suptitle(f'Shear flow')

    st_nSamples, ic_nSamples, kc_nSamples = 2, 3, 3
    constantShear, _ = getSamplesData(dataPath, st_nSamples, ic_nSamples)

    (x_st, s_st, v_st,
     x_ic, s_ic, v_ic,
     x_kc, s_kc, v_kc) = (
        constantShear['st_time'],
        constantShear['st_stressCte'],
        constantShear['st_viscosityCte'],
        #
        constantShear['ic_time'],
        constantShear['ic_stressCte'],
        constantShear['ic_viscosityCte'],
        #
        constantShear['kc_time'],
        constantShear['kc_stressCte'],
        constantShear['kc_viscosityCte'])

    table = []

    table = plotFlowTime(
        listRows=table, nSamples=st_nSamples,
        ax=axStress, x=x_st, y=s_st,
        axTitle='', yLabel='Shear stress (Pa)', yLim=(100, 600),
        curveColor='silver', markerStyle='o',
        sampleName=f'10_0WSt')

    table = plotFlowTime(
        listRows=table, nSamples=ic_nSamples,
        ax=axStress, x=x_ic, y=s_ic,
        axTitle='', yLabel='Shear stress (Pa)', yLim=(100, 600),
        curveColor='deepskyblue', markerStyle='o',
        sampleName=f'10_0WSt_iCar')

    table = plotFlowTime(
        listRows=table, nSamples=kc_nSamples,
        ax=axStress, x=x_kc, y=s_kc,
        axTitle='', yLabel='Shear stress (Pa)', yLim=(100, 600),
        curveColor='navajowhite', markerStyle='o',
        sampleName=f'10_0WSt_kCar')

    # plt.subplots_adjust(wspace=0.175, top=0.890, bottom=0.14, left=0.05, right=0.95)
    plt.tight_layout()
    plt.show()
    fig.savefig(f'{dirSave}' + f'\\{fileName}' + '.png', facecolor='w', dpi=600)

    fitParams = pd.DataFrame(table)
    fitParams.to_excel(f'{dirSave}' + f'\\{fileName}' + '.xlsx', index=False)

    print(f'\n\n· Chart and table with fitted parameters saved at\n{dirSave}.')


if __name__ == '__main__':
    # folderPath = "C:/Users/petrus.kirsten/PycharmProjects/RheometerPlots/data"
    folderPath = "C:/Users/Petrus Kirsten/Documents/GitHub/RheometerPlots/data"
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
        folderPath + "/091024/10_0WSt_kCar/10_0WSt_kCar-viscoelasticRecovery-Flow_4a.xlsx",
    ]

    main(dataPath=filePath)
