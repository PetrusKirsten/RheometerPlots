import os
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
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
        save,
        plotError,  # plot std. dev. as error in spectra
        wavelength,  # in nm
        figSize=(24, 18),
        dpi=100
):
    cm = 1 / 2.54  # cm in inches

    # Data reading
    data = pd.read_csv(data_path)

    wavelengthArray = data.iloc[0, 1:].to_numpy()

    s1 = data[data['Content'] == 'S1'].iloc[:, 1:].to_numpy()
    s1 = s1[~np.isnan(s1)]

    s2 = data[data['Content'] == 'S2'].iloc[:, 1:].to_numpy()
    s2 = s2[~np.isnan(s2)]

    s3 = data[data['Content'] == 'S3'].iloc[:, 1:].to_numpy()
    s3 = s3[~np.isnan(s3)]

    s4 = data[data['Content'] == 'S4'].iloc[:, 1:].to_numpy()
    s4 = s4[~np.isnan(s4)]

    s5 = data[data['Content'] == 'S5'].iloc[:, 1:].to_numpy()
    s5 = s5[~np.isnan(s5)]

    s6 = data[data['Content'] == 'S6'].iloc[:, 1:].to_numpy()
    s6 = s6[~np.isnan(s6)]

    c1 = data[data['Content'] == 'C1'].iloc[:, 1:].to_numpy()
    c1 = c1[~np.isnan(c1)]

    c2 = data[data['Content'] == 'C2'].iloc[:, 1:].to_numpy()
    c2 = c2[~np.isnan(c2)]

    c3 = data[data['Content'] == 'C3'].iloc[:, 1:].to_numpy()
    c3 = c3[~np.isnan(c3)]

    c4 = data[data['Content'] == 'C4'].iloc[:, 1:].to_numpy()
    c4 = c4[~np.isnan(c4)]

    # Remove outliers based on a previous observation
    # s2 = np.delete(s2, 0, 0)
    # c1 = np.delete(c1, 1, 0)
    # c4 = np.delete(c4, 0, 0)

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
    wl = wavelength  # nm
    absWL = dict()
    errWL = dict()

    for sample in valuesMeans:
        inAbsWL = list()
        inAbsWL.append(valuesMeansNBC[sample][0])  # append the label to the list
        inAbsWL.append(valuesMeansNBC[sample][1])  # append the value to the list
        inAbsWL.append(errors[sample][1])  # append the error to the list

        absWL[sample] = inAbsWL

    plt.style.use('seaborn-v0_8-ticks')
    fileTitle = os.path.basename(data_path).split("/")[-1].split(".")[0]

    # Linear regression of standard samples
    concs = np.array([2000, 1000, 500, 250, 125, 0])
    npWL = np.array(list(absWL.values()))  # transform the dict to an np.array to
    optimal, covariance = curve_fit(  # access specific axins inside the list value
        linear_reg,
        concs, npWL[:6, 1].astype(float),  # slice first 6 values => standard curve
        p0=(0, 0))
    error = np.sqrt(np.diag(covariance))
    slope = optimal[0]
    stdev = error[0]
    intercept = optimal[1]

    # calculate regression confidence interval
    slopeCI, interceptCI = unc.correlated_values(optimal, covariance)
    py = slopeCI.nominal_value * concs + interceptCI.nominal_value
    nom = unp.nominal_values(py)
    std = unp.std_devs(py)

    figCon, ax = plt.subplots(
        num='Absorbance X [Protein] curve ' + fileTitle,
        figsize=(25 * cm, 20 * cm),
        nrows=1, ncols=1)
    ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.5)
    ax.set_ylabel(f'Absorbance at {wl} nm')
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    # ax.set_ylim([-0.35, 1.2])
    ax.set_xlim([-250, 2250])
    ax.set_xticks(np.arange(-100, 2200, 100))
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_xlabel('Protein concentration (μg/ml)')

    # Standard data
    ax.errorbar(
        concs, npWL[:6, 1].astype(float), yerr=npWL[:6, 2].astype(float),
        fmt='o', markersize=8, alpha=1, color='dodgerblue', markeredgecolor='#383838', markeredgewidth=0.5,
        capsize=3, capthick=3, elinewidth=1, ecolor='dimgrey',
        lw=1, ls='-',
        label='BSA standard', zorder=3)

    # Fitting plot
    concsFit = np.arange(-200, 2200, 1)
    ax.plot(
        concs, linear_reg(concs, slope, intercept),
        lw=1, color='dimgrey', alpha=0.65, zorder=2)
    plt.fill_between(concs, nom - 1.96 * std, nom + 1.96 * std,
                     color='whitesmoke', alpha=0.5, zorder=1)
    # plt.plot(concBSA, nom - 1.96 * std,
    #          c='grey', alpha=0.5, lw=0.75, ls='--', zorder=2)
    # plt.plot(concBSA, nom + 1.96 * std,
    #          c='grey', alpha=0.5, lw=0.75, ls='--',  zorder=2)

    # Collagen data

    colors = ['lightcoral', 'orange', 'mediumseagreen', 'deeppink']
    for dil in range(0, 4):
        # Intercept the Col absorbance values to determine concentration
        concCOL = np.interp(
            npWL[6:, 1].astype(float)[dil],
            linear_reg(concs, slope, intercept),
            concs,
        )
        ax.errorbar(
            concCOL, npWL[6:, 1].astype(float)[dil], yerr=npWL[6:, 2].astype(float)[dil],
            fmt='o', markersize=6, alpha=0.6, color=colors[dil], markeredgecolor='#383838', markeredgewidth=0.5,
            capsize=3, capthick=3, elinewidth=1, ecolor='dimgrey',
            label=f'[{npWL[6:, 0][dil]}] = {concCOL:.1f} μg/ml', zorder=4)
    ax.grid(which='both', alpha=0.9, color='whitesmoke')
    ax.legend(loc='upper right', ncol=1, frameon=False, facecolor='w')

    if save:
        figCon.savefig(f'Calibration curve {fileTitle}.png', dpi=300)


if __name__ == "__main__":
    main('Bradford_300824_PK-2.csv',
         False, True,
         595)
    plt.show()
