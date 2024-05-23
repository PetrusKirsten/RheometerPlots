import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties


# Fonts in chart config
def fonts(folder_path, small=10, medium=12, big=14):
    font_path = folder_path + 'HelveticaNeueThin.otf'
    helvetica_thin = FontProperties(fname=font_path)

    font_path = folder_path + 'HelveticaNeueLight.otf'
    helvetica_light = FontProperties(fname=font_path)

    font_path = folder_path + 'HelveticaNeueMedium.otf'
    helvetica_medium = FontProperties(fname=font_path)

    font_path = folder_path + 'HelveticaNeueBold.otf'
    helvetica_bold = FontProperties(fname=font_path)

    plt.rc('font', size=small)  # controls default text sizes
    plt.rc('axes', titlesize=medium)  # fontsize of the axes title
    plt.rc('axes', labelsize=medium)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=medium)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=medium)  # fontsize of the tick labels
    plt.rc('legend', fontsize=medium)  # legend fontsize
    plt.rc('figure', titlesize=medium)  # fontsize of the figure title


# Global configs
np.set_printoptions(threshold=np.inf)  # print the entire array
cm = 1 / 2.54  # centimeters in inches
fonts(folder_path='C:/Users/petrus.kirsten/AppData/Local/Microsoft/Windows/Fonts/')


def linear_reg(x, slope, inter):
    return slope * x + inter


def main(data_path, strain_plot=True):
    # If strain_plot is False, it shows the Stress X Time plot

    # Data config
    df = pd.read_csv(data_path)
    # print(df.to_string())  # To print all the values from dataframe

    max_id = df['Fn in N'].idxmax()  # index of max force value
    max_h = df['h in mm'].iloc[max_id]  # find the height where the force is max

    array_df = df.drop(columns=['SegIndex']).to_numpy()
    # print(array_df[:, 3])  # select values from 3rd column

    pts_period = 196  # Data points per oscillation/period
    n_cycles = int(array_df[:, 3].shape[0] / pts_period)  # Number of oscillations/periods
    sample_area = np.pi * 0.015 ** 2 * 1000  # 30 mm diamater circle * kilo

    t_seq = array_df[:, 3].reshape(n_cycles, pts_period)
    half_period = int(t_seq.shape[1] / 2)
    fn_seq = array_df[:, 0].reshape(n_cycles, pts_period)
    str_seq = fn_seq / sample_area
    h_seq = array_df[:, 1].reshape(n_cycles, pts_period)

    t_total = array_df[:, 2]
    fn_total = array_df[:, 0]
    h_total = array_df[:, 1] - array_df[:, 1].min()

    # Get the max stress values from a range and store the means and std dev
    peak_mean = np.array([])
    peak_std = np.array([])
    maxs_index = np.argmax(fn_seq, axis=1)
    for f in range(len(maxs_index)):
        ran = 5
        left = str_seq[f, maxs_index[f] - ran:maxs_index[f]]
        right = str_seq[f, maxs_index[f]:maxs_index[f] + ran]
        peak = np.append(left, right)
        peak_mean = np.append(peak_mean, peak.mean())
        peak_std = np.append(peak_std, peak.std())

    # maxs_forces = np.max(np.max(fn_seq, axis=1).reshape(3, 2), axis=1)

    # Geral plots configs
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(34 * cm, 15 * cm),
        gridspec_kw={'width_ratios': [3, 2]})

    color_series = 'royalblue'
    color_linrange = 'firebrick'

    # Left plot configs
    ax1.set_xlim([0, 2])
    if strain_plot:  # If False, it shows the Stress X Time plot
        ax1.set_xlabel('Strain')
        ax1.set_xticks([0, 0.5, 1, 1.5, 2])
        ax1.set_xticklabels(['0%', '10%', '20%', '-10%', '-20%'])
        ax1.xaxis.set_minor_locator(MultipleLocator(0.25))
    else:
        ax1.set_xlabel('Oscillation time (s)')

    ax1.set_ylabel('Stress (kPa)')
    ax1.yaxis.set_major_locator(MultipleLocator(0.5))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax1.set_ylim([-0.25, str_seq.max() + str_seq.max()*0.15])

    strain_i = t_seq[0, 40]
    strain_f = t_seq[0, 90]

    ax1.axvspan(strain_i, strain_f, alpha=0.25, color='lightcoral')

    for i in np.arange(0, n_cycles, 1):
        popt, pcov = curve_fit(
            linear_reg,
            t_seq[i, 40:90],
            str_seq[i, 40:90],
            p0=(2, 0)
        )

        ax1.scatter(
            np.append(t_seq[i, :half_period], t_seq[i, half_period:] + 1), str_seq[i, :],
            label=f'#{i + 1} period', color=color_series, edgecolors='none', s=30, alpha=(0.85 - 0.25 * i))
        # ax1.scatter([t_seq[i, 40], t_seq[i, 90]], [str_seq[i, 40], str_seq[i, 90]], color='darkslategrey', alpha=0.75)

    ax1.text(strain_i + 0.08, str_seq.max() + str_seq.max()*0.1, f'Linear region', color=color_linrange, weight='bold')

    ax1.text(strain_i + 0.02, -0.1, f'{strain_i * 20:.1f}%', color=color_linrange)
    ax1.text(strain_f - 0.17, -0.1, f'{strain_f * 20:.1f}%', color=color_linrange)
    ax1.legend(frameon=False)

    # Right upper plot configs
    period_array = np.arange(1, n_cycles + 1, 1)
    ax2.set_xlabel('Period')
    ax2.set_xticks(period_array)
    ax2.set_xlim([period_array[0] - 0.7, period_array[-1] + 0.7])
    ax2.set_ylabel('Peak stress per period (kPa)')
    ax2.set_ylim([peak_mean.min() - peak_mean.min() * 0.05, peak_mean.max() + peak_mean.max() * 0.05])

    ax2.errorbar(
        period_array, peak_mean, yerr=peak_std, alpha=1,
        fmt='o', markersize=9, color=color_series, markeredgecolor='#383838', markeredgewidth=1,
        capsize=3, capthick=1, elinewidth=1, ecolor='#383838', linestyle='-')

    for val in period_array:  # Show the mean values and the std dev of each peak
        # X of text a lil bit from the left of the center period
        # Y of the test in the center (peak_mean[val - 1])
        # of upper error bar (peak_std[val - 1] + peak_std[val - 1] * 0.1)
        ax2.text(val - 0.37,
                 peak_mean[val - 1] + peak_std[val - 1] + peak_std[val - 1] * 0.15,
                 f'{peak_mean[val - 1]:.2f} ± {peak_std[val - 1]:.2f} kPa',
                 color='#383838')
    ax2.text(period_array[0] - 0.6, peak_mean.min() - peak_mean.min() * 0.045,
             f'Decrease in stress peak: '
             f'{abs(100 - (peak_mean[0] / peak_mean[-1]) * 100):.1f}%', color='#383838')

    # Right lower plot configs
    period_array = np.arange(1, n_cycles + 1, 1)
    ax2.set_xlabel('Period')
    ax2.set_xticks(period_array)
    ax2.set_xlim([period_array[0] - 0.7, period_array[-1] + 0.7])
    ax2.set_ylabel('Peak stress per period (kPa)')
    ax2.set_ylim([peak_mean.min() - peak_mean.min() * 0.05, peak_mean.max() + peak_mean.max() * 0.05])

    ax2.errorbar(
        period_array, peak_mean, yerr=peak_std, alpha=1,
        fmt='o', markersize=9, color=color_series, markeredgecolor='#383838', markeredgewidth=1,
        capsize=3, capthick=1, elinewidth=1, ecolor='#383838', linestyle='-')

    for val in period_array:  # Show the mean values and the std dev of each peak
        # X of text a lil bit from the left of the center period
        # Y of the test in the center (peak_mean[val - 1])
        # of upper error bar (peak_std[val - 1] + peak_std[val - 1] * 0.1)
        ax2.text(val - 0.37,
                 peak_mean[val - 1] + peak_std[val - 1] + peak_std[val - 1] * 0.15,
                 f'{peak_mean[val - 1]:.2f} ± {peak_std[val - 1]:.2f} kPa',
                 color='#383838')
    ax2.text(period_array[0] - 0.6, peak_mean.min() - peak_mean.min() * 0.045,
             f'Decrease in stress peak: '
             f'{abs(100 - (peak_mean[0] / peak_mean[-1]) * 100):.1f}%', color='#383838')


if __name__ == "__main__":
    main('../data/haribo-9v2.csv')

    plt.show()
