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
        plotError,  # plot std. dev. as error in spectra
        wavelenth,  # in nm
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

    # Remove outliers based on a previous observation
    s2 = np.delete(s2, 0, 0)
    c1 = np.delete(c1, 1, 0)
    c4 = np.delete(c4, 0, 0)

    valuesNBC = {1: ['2.0 mg/ml', s1],  # NBC: "no blank correction"
                 2: ['1.0 mg/ml', s2],
                 3: ['0.5 mg/ml', s3],
                 4: ['0.250 mg/ml', s4],
                 5: ['0.125 mg/ml', s5],
                 6: ['0 mg/ml - Blank', s6],
                 7: ['1× ASCol-II', c1],
                 8: ['10× ASCol-II', c2],
                 9: ['50× ASCol-II', c3],
                 10: ['100× ASCol-II', c4]}

    valuesMeansNBC = {1: ['2.0 mg/ml', np.mean(s1, axis=0)],  # NBC: "no blank correction"
                      2: ['1.0 mg/ml', np.mean(s2, axis=0)],
                      3: ['0.5 mg/ml', np.mean(s3, axis=0)],
                      4: ['0.250 mg/ml', np.mean(s4, axis=0)],
                      5: ['0.125 mg/ml', np.mean(s5, axis=0)],
                      6: ['0 mg/ml - Blank', np.mean(s6, axis=0)],
                      7: ['1× ASCol-II', np.mean(c1, axis=0)],
                      8: ['10× ASCol-II', np.mean(c2, axis=0)],
                      9: ['50× ASCol-II', np.mean(c3, axis=0)],
                      10: ['100× ASCol-II', np.mean(c4, axis=0)]}

    valuesMeans = {1: ['S1', np.mean(s1, axis=0) - np.mean(s6, axis=0)],
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

    # Select a specific wavelenth (wl) to analyze the spectra
    wl = wavelenth  # nm
    indexWL = np.where(data.iloc[0, :].to_numpy() == wl)[0][0]  # Find index of a specific wavelenght
    absWL = list()
    for sample in valuesMeans:
        inAbsWL = valuesMeans[sample][1][indexWL]  # Find the index of a spec. wl in each spectrum
        absWL.append(inAbsWL)                      # append to a new list

    plt.style.use('seaborn-v0_8-ticks')
    fileTitle = os.path.basename(data_path).split("/")[-1].split(".")[0]

    # Figure configurations for all plots together.
    fig = plt.figure(
        'Spectra together ' + fileTitle,
        figsize=(figSize[0] * cm, figSize[1] * cm),
        dpi=dpi)
    fig.suptitle(f'{fileTitle}', alpha=0.9)
    fig.subplots_adjust(hspace=0)
    gs = GridSpec(1, 1)
    ax = fig.add_subplot(gs[:, 0])

    # ax.set_xlabel('Wavelength (μg/ml)')
    # ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    # ax.set_xticks([0, 0.5, 1, 1.5, 2])

    ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.5)
    ax.set_xlim([400, 700])
    ax.set_ylim([-0.35, 1.2])
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Absorbance')

    # Spectrum plot
    for curve in range(1, 11):
        init = 0.9
        dif = 0.1
        lineColor = 'dodgerblue'

        if curve == 6:
            lineColor = 'grey'
        elif curve > 6:
            init = 2.2
            dif = 0.2
            lineColor = 'mediumvioletred'

        ax.plot(
            wavelength, valuesMeans[curve][1],
            color=lineColor, alpha=init - dif * curve, lw=1.25,
            label=valuesMeansNBC[curve][0], zorder=2)
        if plotError:
            plt.fill_between(
                wavelength.tolist(),
                (valuesMeans[curve][1] + errors[curve][1]).tolist(),
                (valuesMeans[curve][1] - errors[curve][1]).tolist(),
                color=lineColor, alpha=0.10, zorder=1)

    # Wavelenght line markers
    plt.axvline(x=wl, color='orange', alpha=0.75, ls='-', lw=0.85)
    plt.text(wl, 1.1, f'{wl} nm',
             horizontalalignment='center',
             verticalalignment='center',
             color='darkorange', backgroundcolor='w', alpha=1)
    plt.axvline(x=474, color='orange', alpha=0.75, ls='-', lw=0.85)
    plt.text(474, 0.5, f'474 nm',
             horizontalalignment='center',
             verticalalignment='center',
             color='darkorange', backgroundcolor='w', alpha=1)
    ax.legend(loc=2, ncol=2, frameon=False, facecolor='w')

    # Figure configurations for separeted spectra
    fig2, axes = plt.subplots(
        num='Separated plots ' + fileTitle,
        figsize=(40 * cm, 20 * cm),
        nrows=2, ncols=5,
        sharex=True, sharey=True)

    fig2.patch.set_facecolor('whitesmoke')
    fig2.suptitle(f'{fileTitle}', alpha=0.9)
    fig2.subplots_adjust(hspace=0.05, wspace=0.1)

    axes[1, 0].set_xlabel('Wavelength (nm)')
    axes[1, 0].set_ylabel('Absorbance')

    line = 1                          # interactor for each samples curve
    for r in np.arange(0, 2, 1):      # loop for row of samples spectra
        for c in np.arange(0, 5, 1):  # loop for columns of samples spectra
            ax = axes[r, c]

            ax.set_title(f'{valuesNBC[line][0]}', size=9)
            ax.spines[['left', 'right', 'top', 'bottom']].set_linewidth(0.5)
            ax.tick_params(direction='in', length=3, width=0.5, colors='k',
                           grid_color='grey', grid_alpha=0.5)
            ax.set_xlim([400, 700])
            ax.set_ylim([0.2, 1.4])
            ax.yaxis.set_tick_params(labelbottom=False)
            ax.set_yticks([])

            ax.plot(wavelength, valuesNBC[line][1][0], c='dodgerblue', lw=1.2, label=1)
            ax.plot(wavelength, valuesNBC[line][1][1], c='hotpink', lw=1.2, label=2)
            ax.plot(wavelength, valuesNBC[line][1][-1], c='orange', lw=1.2, label=3)

            # Wavelenght line markers
            ax.axvline(x=wl, color='grey', alpha=0.75, ls='--', lw=0.85)
            ax.text(wl, 1.1, f'{wl} nm',
                    horizontalalignment='center', verticalalignment='center',
                    size=7, color='dimgrey', backgroundcolor='w', alpha=1)
            ax.axvline(x=474, color='grey', alpha=0.75, ls='--', lw=0.85)
            ax.text(474, 1.1, f'474 nm',
                    horizontalalignment='center', verticalalignment='center',
                    size=7, color='dimgrey', backgroundcolor='w', alpha=1)
            ax.legend(loc=2, ncol=1, frameon=False, facecolor='w')
            line += 1

    plt.tight_layout()
    plt.show()

    fig.savefig(f'Individual spectra {fileTitle}.png', dpi=300)
    fig2.savefig(f'Spectra together {fileTitle}.png', dpi=300)

    # TODO: fazer gráfico ABS x [Protein] com regressão e estimar resultados

    # Linear regression of standard samples
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

    # Standard data ???
    ax.errorbar(
        wavelength, absWL, yerr=absWL,
        fmt='o', markersize=6.5,  alpha=1, color='lightgrey', markeredgecolor='#383838', markeredgewidth=0.5,
        capsize=3, capthick=3, elinewidth=1, ecolor='darkgrey', zorder=3,
        label='BSA standard')

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


if __name__ == "__main__":
    main('bradford_ascol-II_030924.csv', False, 595)
