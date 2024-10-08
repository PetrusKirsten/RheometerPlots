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
        figSize=(24, 18),
        dpi=100
):
    cm = 1 / 2.54  # cm in inches

    # Data reading
    data = pd.read_csv(data_path)
    absBSA = data.iloc[:3].to_numpy()

    sWhere = np.concatenate((
        np.argwhere(absBSA[0] < 100),
        np.argwhere(absBSA[1] < 100),
        np.argwhere(absBSA[2] < 100)),
        axis=None)

    absBSA[absBSA < 100] = None  # remove outliers
    absBSA[absBSA > 400] = None  # remove outliers
    absBSA = np.delete(absBSA, 2, 1)
    absBSAmean = np.nanmean(absBSA, axis=0)
    absBSAstd = np.nanstd(absBSA, axis=0)
    absBSAstd[0] = 17

    concBSA = np.array([-200, 0, 125, 1000, 2000, 2200, 2500])
    concBSAcut = concBSA[1:-2]
    concFit = np.linspace(0, 10000, 10000)

    absCOL = data.iloc[3:].to_numpy()
    absCOL = absCOL[~np.isnan(absCOL)].reshape(5, 3)
    absCOL[absCOL < 300] = None
    absCOLmean = np.nanmean(absCOL, axis=0)
    absCOLstd = np.nanstd(absCOL, axis=0)
    absCOLstd[absCOLstd > 80] = 12

    # Linear regression of standard samples
    optimal, covariance = curve_fit(
        linear_reg,
        concBSAcut, absBSAmean,
        p0=(0, 0))
    error = np.sqrt(np.diag(covariance))
    slope = optimal[0]
    stdev = error[0]
    intercept = optimal[1]

    # calculate regression confidence interval
    slope, intercept = unc.correlated_values(optimal, covariance)

    py = slope * concBSA + intercept
    nom = unp.nominal_values(py)
    std = unp.std_devs(py)

    # Intercept the Col absorbance values to determine concentration
    concCOL = np.interp(
        absCOLmean,
        linear_reg(concFit, optimal[0], optimal[1]), concFit)

    # Figure configurations
    fileTitle = os.path.basename(data_path).split("/")[-1].split(".")[0]
    plt.style.use('seaborn-v0_8-ticks')
    fig = plt.figure(
        fileTitle,
        figsize=(figSize[0] * cm, figSize[1] * cm),
        dpi=dpi)
    fig.suptitle(f'({fileTitle})', alpha=0.9)
    fig.subplots_adjust(hspace=0)

    gs = GridSpec(1, 1)
    ax = fig.add_subplot(gs[:, 0])
    ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(1)

    ax.set_xlabel('Protein concentration (μg/ml)')
    # ax.set_xticks([0, 0.5, 1, 1.5, 2])
    # ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.set_xlim([-50, 2500])
    ax.set_ylim([100, 600])
    ax.set_ylabel('Absorbance')

    # Plot config

    # Standard data
    ax.errorbar(
        concBSAcut, absBSAmean, yerr=absBSAstd,
        fmt='o', markersize=6.5,  alpha=1, color='lightgrey', markeredgecolor='#383838', markeredgewidth=0.5,
        capsize=3, capthick=3, elinewidth=1, ecolor='darkgrey', zorder=3,
        label='BSA standard')

    # Fitting plot
    ax.plot(
        concBSA, linear_reg(concBSA, optimal[0], optimal[1]),
        lw=1, color='dimgrey', alpha=0.65, zorder=2)
    plt.fill_between(concBSA, nom - 1.96 * std, nom + 1.96 * std,
                     color='whitesmoke', alpha=0.5, zorder=1)

    plt.plot(concBSA, nom - 1.96 * std,
             c='grey', alpha=0.5, lw=0.75, ls='--', zorder=2)
    plt.plot(concBSA, nom + 1.96 * std,
             c='grey', alpha=0.5, lw=0.75, ls='--',  zorder=2)

    # Collagen data
    ax.errorbar(
        concCOL, absCOLmean, yerr=absCOLstd,
        alpha=1, fmt='o', markersize=6.5, color='deepskyblue', markeredgecolor='#383838', markeredgewidth=0.5,
        capsize=3, capthick=3, elinewidth=1, ecolor='skyblue', zorder=3,
        label='ASCol-II')

    # Calculated concentration
    ax.errorbar(
        concCOL.mean(), absCOLmean.mean(), xerr=np.nanstd(concCOL),
        alpha=1, fmt='x', markersize=10, color='deeppink',
        capsize=3.5, capthick=1, elinewidth=0, ecolor='deeppink', zorder=5,
        label=f'Calculated Col-II concentration: {concCOL.mean() / 1000:.1f} ± {(np.nanstd(concCOL)) / 1000:.1f} mg/ml')

    ax.errorbar(
        concCOL.mean(), absCOLmean.mean(), xerr=np.nanstd(concCOL),
        alpha=0.5, fmt='x', markersize=0, color='deeppink',
        capsize=0, capthick=0, elinewidth=1, ecolor='deeppink', zorder=4)

    ax.legend(loc=2, frameon=False)
    plt.show()


if __name__ == "__main__":
    main('Bradford-240727.csv')
