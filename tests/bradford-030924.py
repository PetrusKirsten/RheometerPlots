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

    wavelength = data.iloc[0, 1:].to_numpy()

    s1 = data[data['Content'] == 'Standard S1'].iloc[:, 1:].to_numpy()
    s2 = data[data['Content'] == 'Standard S2'].iloc[:, 1:].to_numpy()
    s3 = data[data['Content'] == 'Standard S3'].iloc[:, 1:].to_numpy()
    s4 = data[data['Content'] == 'Standard S4'].iloc[:, 1:].to_numpy()
    s5 = data[data['Content'] == 'Standard S5'].iloc[:, 1:].to_numpy()
    s6 = data[data['Content'] == 'Standard S6'].iloc[:, 1:].to_numpy()

    c1 = data[data['Content'] == 'Sample X1'].iloc[:, 1:].to_numpy()
    c2 = data[data['Content'] == 'Sample X2'].iloc[:, 1:].to_numpy()
    c3 = data[data['Content'] == 'Sample X3'].iloc[:, 1:].to_numpy()
    c4 = data[data['Content'] == 'Sample X10'].iloc[:, 1:].to_numpy()

    means = {1: ['S1', np.mean(s1, axis=0) - np.mean(s6, axis=0)],
             2: ['S2', np.mean(s2, axis=0) - np.mean(s6, axis=0)],
             3: ['S3', np.mean(s3, axis=0) - np.mean(s6, axis=0)],
             4: ['S4', np.mean(s4, axis=0) - np.mean(s6, axis=0)],
             5: ['S5', np.mean(s5, axis=0) - np.mean(s6, axis=0)],
             6: ['S6', np.mean(s6, axis=0) - np.mean(s6, axis=0)],
             7: ['C1', np.mean(c1, axis=0) - np.mean(s6, axis=0)],
             8: ['C2', np.mean(c2, axis=0) - np.mean(s6, axis=0)],
             9: ['C3', np.mean(c3, axis=0) - np.mean(s6, axis=0)],
             10: ['C4', np.mean(c4, axis=0) - np.mean(s6, axis=0)]}

    errors = {1: ['S1', np.std(s1, axis=0)],
              2: ['S2', np.std(s2, axis=0)],
              3: ['S3', np.std(s3, axis=0)],
              4: ['S4', np.std(s4, axis=0)],
              5: ['S5', np.std(s5, axis=0)],
              6: ['S6', np.std(s6, axis=0)],
              7: ['C1', np.std(c1, axis=0)],
              8: ['C2', np.std(c2, axis=0)],
              9: ['C3', np.std(c3, axis=0)],
              10: ['C4', np.std(c4, axis=0)]}

    # abs_std_1 = data.iloc[:, 1:7]
    # abs_std_2 = data.iloc[:, 11:17]
    # abs_std_3 = data.iloc[:, 21:27]
    #
    # abs_col_1 = data.iloc[:, 7:11]
    # abs_col_2 = data.iloc[:, 17:21]
    # abs_col_3 = data.iloc[:, 27:29]
    #
    # abs_col_12 = data.iloc[:, 29:]

    # Linear regression of standard samples
    # optimal, covariance = curve_fit(
    #     linear_reg,
    #     concBSAcut, absBSAmean,
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
    fig.suptitle(f'({fileTitle})', alpha=0.9)
    fig.subplots_adjust(hspace=0)

    gs = GridSpec(1, 1)
    ax = fig.add_subplot(gs[:, 0])
    ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(1)

    ax.set_xlabel('Wavelength (μg/ml)')
    # ax.set_xticks([0, 0.5, 1, 1.5, 2])
    # ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.set_xlim([400, 700])
    # ax.set_ylim([0.2, 1.4])
    ax.set_ylabel('Absorbance')

    # Plot config
    # np.mean(s1, axis=0) + np.std(s1, axis=0)
    # np.mean(s1, axis=0) - np.std(s1, axis=0)
    # ax.plot(
    #     wavelength, np.mean(s1, axis=0) - np.std(s1, axis=0),
    #     color='dodgerblue', alpha=0.25, zorder=1)
    # ax.plot(
    #     wavelength, np.mean(s1, axis=0) + np.std(s1, axis=0),
    #     color='dodgerblue', alpha=0.25, zorder=1)

    # Spectrum plot
    for curve in range(1, 7):
        ax.plot(
            wavelength, means[curve][1],
            color='dodgerblue', alpha=0.9 - 0.1*curve, zorder=2,
            label=means[curve][0])
        # plt.fill_between(
        #     wavelength.tolist(),
        #     (means[curve][1] + errors[curve][1]).tolist(),
        #     (means[curve][1] - errors[curve][1]).tolist(),
        #     color='dodgerblue', alpha=0.10, zorder=1)

    for curve in range(7, 11):
        ax.plot(
            wavelength, means[curve][1],
            color='mediumvioletred', alpha=2.2 - 0.2*curve, zorder=2,
            label=means[curve][0])
        # plt.fill_between(
        #     wavelength.tolist(),
        #     (np.mean(s1, axis=0) + np.std(s1, axis=0)).tolist(),
        #     (np.mean(s1, axis=0) - np.std(s1, axis=0)).tolist(),
        #     color='mediumvioletred', alpha=0.10, zorder=1)

    plt.axvline(x=595, color='springgreen', alpha=0.75, ls='-', lw=0.85, label='595 nm')

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

    ax.legend(loc=2, frameon=False)
    plt.show()


if __name__ == "__main__":
    main('bradford_ascol-II_030924.csv')
