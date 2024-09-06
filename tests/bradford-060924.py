import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

# pip install uncertainties, if needed
try:
    import uncertainties.unumpy as unp
    import uncertainties as unc
except:
    try:
        from pip import main as pipmain
    except:
        from pip._internal import main as pipmain
    pipmain(['install', 'uncertainties'])
    import uncertainties.unumpy as unp
    import uncertainties as unc


def linear_reg(x, slope, interc):
    return slope * x + interc


def predband(x, xd, yd, p, func, conf=0.95):
    # x = requested points
    # xd = x data
    # yd = y data
    # p = parameters
    # func = function name
    alpha = 1.0 - conf  # significance
    N = xd.size  # data sample size
    var_n = len(p)  # number of parameters
    # Quantile of Student's t distribution for p=(1-alpha/2)
    q = stats.t.ppf(1.0 - alpha / 2.0, N - var_n)
    # Stdev of an individual measurement
    se = np.sqrt(1. / (N - var_n) * np.sum((yd - func(xd, *p)) ** 2))
    # Auxiliary definitions
    sx = (x - xd.mean()) ** 2
    sxd = np.sum((xd - xd.mean()) ** 2)
    # Predicted values (best-fit model)
    yp = func(x, *p)
    # Prediction band
    dy = q * se * np.sqrt(1.0 + (1.0 / N) + (sx / sxd))
    # Upper & lower prediction bands.
    lpb, upb = yp - dy, yp + dy

    return lpb, upb


def main(
        data_path,
        plotError,
        figSize=(24, 18),
        dpi=100
):
    cm = 1 / 2.54  # cm in inches

    # Data reading
    data = pd.read_csv(data_path)

    wavelength = np.arange(400, 700 + 5, 5)

    # correction
    dataAll = list()
    for r in np.arange(0, 30, 1):
        for c in np.arange(1, 123, 2):
            pre = str(data.iloc[r, c])       # before .
            post = str(data.iloc[r, c + 1])  # after .
            dataAll.append(float(pre + '.' + post))
    dataAll = np.array(dataAll).reshape(30, 61)

    labels = data['Content'].to_numpy().reshape(10, 3)[:, 0]
    dataMeans = dict()
    errors = dict()
    s = 0
    c = 1
    for key in labels:
        mean = np.mean(dataAll[s:s + 3], axis=0)
        dataMeans[c] = [key, mean]
        error = np.std(dataAll[s:s + 3], axis=0)
        errors[c] = [key, error]
        s += 3
        c += 1

    dataMeansBC = dict()  # BC: "blank correction"
    s = 0
    c = 1
    for key in labels:
        meanNBC = np.mean(dataAll[s:s + 3], axis=0) - dataMeans[10][1]
        dataMeansBC[c] = [key, meanNBC]
        s += 3
        c += 1

    wl = 595  # nm
    # indexWL = np.where(data.iloc[0, :].to_numpy() == wl)[0][0]  # Find index of a specific wavelenght
    # absWL = list()
    #
    # for sample in means:
    #     inAbsWL = means[sample][1][indexWL]
    #     absWL.append(inAbsWL)

    # Linear regression of standard samples  # TODO: fazer gráfico ABS x [Protein] com regressão e estimar resultados
    # optimal, covariance = curve_fit(
    #     linear_reg,
    #     wavelength, absWL,
    #     p0=(0, 0))
    # error = np.sqrt(np.diag(covariance))
    # slope = optimal[0]
    # stdev = error[0]
    # intercept = optimal[1]

    # calculate regression confidence interval
    # slope, intercept = unc.correlated_values(optimal, covariance)
    #
    # py = slope * concBSA + intercept
    # nom = unp.nominal_values(py)
    # std = unp.std_devs(py)

    # Intercept the Col absorbance values to determine concentration
    # concCOL = np.interp(
    #     absCOLmean,
    #     linear_reg(concFit, optimal[0], optimal[1]), concFit)

    # Figure configurations
    fileTitle = os.path.basename(data_path).split("/")[-1].split(".")[0]
    plt.style.use('seaborn-v0_8-ticks')
    fig = plt.figure(
        fileTitle,
        figsize=(figSize[0] * cm, figSize[1] * cm),
        dpi=dpi)
    fig.suptitle(f'{fileTitle}', alpha=0.9)
    fig.subplots_adjust(hspace=0)

    gs = GridSpec(1, 1)
    ax = fig.add_subplot(gs[:, 0])
    ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(1)

    ax.set_xlabel('Wavelength (nm)')
    # ax.set_xlabel('Wavelength (μg/ml)')
    # ax.set_xticks([0, 0.5, 1, 1.5, 2])
    # ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.set_xlim([400, 700])
    # ax.set_ylim([-0.35, 1.2])
    ax.set_ylabel('Absorbance')

    # TODO: plotar cada medida do espectro individualmente para procurar por outliers
    # Spectrum plot
    for curve in range(1, 2):
        init = 0.9
        dif = 0.1
        lineColor = 'dodgerblue'

        if curve > 6:
            init = 2.2
            dif = 0.2
            lineColor = 'mediumvioletred'

        ax.plot(
            wavelength, dataMeansBC[curve][1],
            color=lineColor, alpha=init - dif * curve, zorder=2,
            label=dataMeans[curve][0])

        if plotError:
            plt.fill_between(
                wavelength.tolist(),
                (dataMeansBC[curve][1] + errors[curve][1]).tolist(),
                (dataMeansBC[curve][1] - errors[curve][1]).tolist(),
                color=lineColor, alpha=0.10, zorder=1)

    # Wavelenght line marker
    plt.axvline(x=wl, color='orange', alpha=0.75, ls='-', lw=0.85)
    plt.text(wl, 1.1, f'{wl} nm',
             horizontalalignment='center',
             verticalalignment='center',
             color='darkorange', backgroundcolor='w', alpha=1)

    # Standard data
    # ax.errorbar(
    #     concBSAcut, absBSAmean, yerr=absBSAstd,
    #     fmt='o', markersize=6.5,  alpha=1, color='lightgrey', markeredgecolor='#383838', markeredgewidth=0.5,
    #     capsize=3, capthick=3, elinewidth=1, ecolor='darkgrey', zorder=3,
    #     label='BSA standard')

    # Fitting plot
    # ax.plot(
    #     concBSA, linear_reg(concBSA, optimal[0], optimal[1]),
    #     lw=1, color='dimgrey', alpha=0.65, zorder=2)
    # plt.fill_between(concBSA, nom - 1.96 * std, nom + 1.96 * std,
    #                  color='whitesmoke', alpha=0.5, zorder=1)
    #
    # plt.plot(concBSA, nom - 1.96 * std,
    #          c='grey', alpha=0.5, lw=0.75, ls='--', zorder=2)
    # plt.plot(concBSA, nom + 1.96 * std,
    #          c='grey', alpha=0.5, lw=0.75, ls='--',  zorder=2)

    # Collagen data
    # ax.errorbar(
    #     concCOL, absCOLmean, yerr=absCOLstd,
    #     alpha=1, fmt='o', markersize=6.5, color='deepskyblue', markeredgecolor='#383838', markeredgewidth=0.5,
    #     capsize=3, capthick=3, elinewidth=1, ecolor='skyblue', zorder=3,
    #     label='ASCol-II')

    # Calculated concentration
    # ax.errorbar(
    #     concCOL.mean(), absCOLmean.mean(), xerr=np.nanstd(concCOL),
    #     alpha=1, fmt='x', markersize=10, color='deeppink',
    #     capsize=3.5, capthick=1, elinewidth=0, ecolor='deeppink', zorder=5,
    #     label=f'Calculated Col-II concentration: {concCOL.mean() / 1000:.1f} ± {(np.nanstd(concCOL)) / 1000:.1f} mg/ml')
    #
    # ax.errorbar(
    #     concCOL.mean(), absCOLmean.mean(), xerr=np.nanstd(concCOL),
    #     alpha=0.5, fmt='x', markersize=0, color='deeppink',
    #     capsize=0, capthick=0, elinewidth=1, ecolor='deeppink', zorder=4)

    ax.legend(loc=2, ncol=2, frameon=False)
    plt.show()


if __name__ == "__main__":
    main('Bradford_060924_PK.csv', True)
