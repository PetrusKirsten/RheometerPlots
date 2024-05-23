import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
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


def linear_reg(x, slope, interc):
    return slope * x + interc


def main(data_path, plot_peak=False, plot_fit=False, strain_plot=True):
    # If strain_plot is False, it shows the Stress X Time plot

    def data_transform(data, pts):  # Transform the data according to the number of pts from each sequence
        sample_area = np.pi * 0.015 ** 2 * 1000  # 30 mm diamater circle * kilo

        df_array = data.drop(columns=['SegIndex']).to_numpy()

        n = int(df_array[:, 3].shape[0] / pts)  # Number of oscillations/periods

        t_array = np.array([])
        temp_array = df_array[:, 3].reshape(int(n * 2), int(pts / 2))
        for c in np.arange(0, temp_array.shape[0], 2):
            t_array = np.append(
                t_array,
                np.append(temp_array[c], temp_array[c + 1] + temp_array[c][-1]))
        t = t_array.reshape(int(n), int(pts))

        half_n = int(t.shape[1] / 2)

        fn = df_array[:, 0].reshape(n, pts)
        s = fn / sample_area
        h = df_array[:, 1].reshape(n, pts)

        # t_total = df_array[:, 2]
        # fn_total = df_array[:, 0]
        # h_total = df_array[:, 1] - df_array[:, 1].min()
        # print(array_df[:, 3])  # select values from 3rd column
        # max_id = df['Fn in N'].idxmax()  # index of max force value
        # max_h = df['h in mm'].iloc[max_id]  # find the height where the force is max

        return n, half_n, t, fn, s, h

    def stress_peak(size, n):  # Get the max stress values from a range and store the means and std dev
        x = np.array([])
        peak = np.array([])
        mean = np.array([])
        std = np.array([])
        maxs_index = np.argmax(fn_seq, axis=1)
        for f in range(len(maxs_index)):
            x = np.append(x, t_seq[f, maxs_index[f] - size:maxs_index[f] + size * 3])
            peak = np.append(peak, str_seq[f, (maxs_index[f] - size):(maxs_index[f] + size * 3)])
            mean = np.append(mean, peak.mean())
            std = np.append(std, peak.std())

        x = x.reshape(n, int(x.shape[0] / n))
        peak = peak.reshape(n, x.shape[1])
        return x, mean, std, peak

    # Data config
    df = pd.read_csv(data_path)
    # print(df.to_string())  # To print all the values from dataframe

    # Prepare the data
    n_periods, half_period, t_seq, fn_seq, str_seq, h_seq = data_transform(df, 196)

    x_peak, peak_mean, peak_std, peak_val = stress_peak(3, n_periods)

    # maxs_forces = np.max(np.max(fn_seq, axis=1).reshape(3, 2), axis=1)

    # Geral plots configs
    fig = plt.figure(figsize=(34 * cm, 15 * cm))
    gs = GridSpec(2, 2, width_ratios=[3, 2])
    fig.subplots_adjust(hspace=0)

    color_series = 'royalblue'
    color_linrange = 'firebrick'

    # Left plot configs
    ax1 = fig.add_subplot(gs[:, 0])
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
    ax1.set_ylim([-0.25, str_seq.max() + str_seq.max() * 0.15])

    strain_i = t_seq[0, 40]
    strain_f = t_seq[0, 90]

    # Plot the linear range and its values
    ax1.axvspan(strain_i, strain_f, alpha=0.25, color='lightcoral')
    ax1.text(strain_i + 0.08, str_seq.max() + str_seq.max() * 0.1,
             f'Linear region', color=color_linrange, weight='bold')
    ax1.text(strain_i + 0.02, -0.1, f'{strain_i * 20:.1f}%', color=color_linrange)
    ax1.text(strain_f - 0.17, -0.1, f'{strain_f * 20:.1f}%', color=color_linrange)
    # Plot the peak range
    if plot_peak:
        ax1.axvspan(x_peak[-1][0], x_peak[-1][-1], alpha=0.25, color='gold')
        ax1.text(x_peak[-1][-1] + 0.03, str_seq.max() + str_seq.max() * 0.1,
                 f'Peak region', color='goldenrod', weight='bold')

    # Linear regression
    slope_val = np.array([])
    slope_std = np.array([])
    for i in np.arange(0, n_periods, 1):
        popt, pcov = curve_fit(
            linear_reg,
            t_seq[i, 40:90],
            str_seq[i, 40:90],
            p0=(2, 0)
        )
        perr = np.sqrt(np.diag(pcov))

        slope_val = np.append(slope_val, popt[0])
        slope_std = np.append(slope_std, perr[0])

        # Plot experimental data
        ax1.scatter(
            np.append(t_seq[i, :half_period], t_seq[i, half_period:]), str_seq[i, :],
            label=f'#{i + 1} period', color=color_series, edgecolors='none', s=30, alpha=(0.85 - 0.25 * i))

        # If plot_peak is True, plot peak data
        if plot_peak:
            ax1.scatter(
                x_peak[i, :], peak_val[i, :],
                color=color_series, edgecolors='gold', linewidths=0.75, s=30, alpha=1)

        # If plot_fit is True, plot fitted curve
        if plot_fit:
            ax1.plot(
                t_seq[i, 40:90], linear_reg(t_seq[i, 40:90], popt[0], popt[1]),
                color=color_linrange, alpha=(0.75 - 0.12 * i), lw=1.3)
            ax1.scatter(
                [t_seq[i, 40], t_seq[i, 90]], [str_seq[i, 40], str_seq[i, 90]],
                color=color_series, edgecolors=color_linrange, linewidths=0.75, s=30, alpha=1)

    ax1.legend(frameon=False)

    # Right upper plot configs
    ax2 = fig.add_subplot(gs[0, 1])
    period_array = np.arange(1, n_periods + 1, 1)
    ax2.set_xlabel('Period')
    ax2.set_xticks(period_array)
    ax2.set_xlim([period_array[0] - 0.7, period_array[-1] + 0.7])
    ax2.set_ylabel('Peak stress (kPa)')
    ax2.set_ylim([peak_mean.min() - peak_mean.min() * 0.05, peak_mean.max() + peak_mean.max() * 0.05])

    ax2.errorbar(
        period_array, peak_mean, yerr=peak_std, alpha=1,
        fmt='o', markersize=8, color=color_series, markeredgecolor='#383838', markeredgewidth=1,
        capsize=3, capthick=1, elinewidth=1, ecolor='#383838', linestyle='-')

    for val in period_array:  # Plot the mean values and the std dev of each peak
        # X of text a lil bit from the left of the center period
        # Y of the test in the center (peak_mean[val - 1])
        # of upper error bar (peak_std[val - 1] + peak_std[val - 1] * 0.1)
        ax2.text(val - 0.37,
                 peak_mean[val - 1] + peak_std[val - 1] + peak_std[val - 1] * 0.2,
                 f'{peak_mean[val - 1]:.2f} ± {peak_std[val - 1]:.2f} kPa',
                 color='#383838', bbox=dict(facecolor='w', alpha=1, edgecolor='w', pad=0))

    diff = 100 - (peak_mean[0] / peak_mean[-1]) * 100
    diff_s = ''
    if diff > 0:
        diff_s = 'Increased'
    elif diff < 0:
        diff_s = 'Decreased'
    ax2.text(period_array[0] - 0.6, peak_mean.min() - peak_mean.min() * 0.045,
             f'{diff_s} in: {abs(diff):.1f}%', color='#383838')

    # Right lower plot configs
    ax3 = fig.add_subplot(gs[1, 1])
    period_array = np.arange(1, n_periods + 1, 1)
    ax3.set_xlabel('Period')
    ax3.set_xticks(period_array)
    ax3.set_xlim([period_array[0] - 0.7, period_array[-1] + 0.7])
    ax3.set_ylabel("Young's modulus (kPa)")
    ax3.set_ylim([slope_val.min() - slope_val.min() * 0.03, slope_val.max() + slope_val.max() * 0.03])

    ax3.errorbar(
        period_array, slope_val, yerr=slope_std, alpha=1,
        fmt='o', markersize=8, color=color_series, markeredgecolor='#383838', markeredgewidth=1,
        capsize=3, capthick=1, elinewidth=1, ecolor='#383838', linestyle='-')

    for val in period_array:  # Show the mean values and the std dev of each peak
        # X of text a lil bit from the left of the center period
        # Y of the test in the center (peak_mean[val - 1])
        # of upper error bar (peak_std[val - 1] + peak_std[val - 1] * 0.1)
        ax3.text(val - 0.37,
                 slope_val[val - 1] + slope_std[val - 1] + slope_std[val - 1] * 0.7,
                 f'{slope_val[val - 1]:.2f} ± {slope_std[val - 1]:.2f} kPa',
                 color='#383838', bbox=dict(facecolor='w', alpha=1, edgecolor='w', pad=0))

    diff = 100 - (slope_val[0] / slope_val[-1]) * 100
    diff_s = ''
    if diff > 0:
        diff_s = 'Increased'
    elif diff < 0:
        diff_s = 'Decreased'
    ax3.text(period_array[0] - 0.6, slope_val.min() - slope_val.min() * 0.025,
             f'{diff_s} in: {abs(diff):.1f}%', color='#383838')


if __name__ == "__main__":
    main('../data/haribo-9v2.csv', plot_peak=True, plot_fit=True, strain_plot=True)

    plt.show()
