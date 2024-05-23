import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from matplotlib.font_manager import FontProperties


def fonts(folder_path, small=10, medium=12, big=14):  # To config different fonts but it isn't working with these
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
    plt.rc('legend', fontsize=small)  # legend fontsize
    plt.rc('figure', titlesize=medium)  # fontsize of the figure title


def linear_reg(x, slope, interc):
    return slope * x + interc


def damped_sinusoid(t, a, lam, w, phi, y):
    return a * np.exp(lam * t) * np.sin(w * t + phi) + y


def abs_damped_sinusoid(t, a, lam, w, phi):
    return abs(a * np.exp(lam * t) * np.sin(w * t + phi))


def sinusoid(t, a, w, phi, y):
    return a * np.sin(w * t + phi) + y


class DynamicCompression:
    # The data must be in .csv where:
    # col 0: 'SegIndex'
    # col 1: 'Fn in N'
    # col 2: 'h in mm'
    # col 3: 't_seq in s'
    def __init__(
            self,
            data_path,
            points,
            figure_size=(34, 15)
    ):
        self.data_path = data_path
        self.points = points

        self.figure_size = figure_size
        self.fig = plt.figure(figsize=(self.figure_size[0] * cm, self.figure_size[1] * cm))
        self.gs = None

        self.stress = None
        self.peak = None
        self.ym = None

        self.plot_exp_h = None

        self.area = np.pi * 0.015 ** 2  # 30 mm diamater circle => 0.015 m => 0.0007 m²

        self.peak_size = 3

        self.slope_val = np.array([])  # Empty arrays to store the linear fitting
        self.slope_std = np.array([])
        self.i_linreg_pct = 7.5  # Initial strain value for linear region
        self.f_linreg_pct = 18  # Final strain value for linear region
        self.i_linreg = (0.5 / 10) * self.i_linreg_pct  # Convert the strain values to time values
        self.f_linreg = (0.5 / 10) * self.f_linreg_pct  # (0.5 s)/(10 %) × x%
        self.i_index = 0
        self.f_index = 0

        self.data = pd.read_csv(self.data_path)
        self.fig.subplots_adjust(hspace=0)

        # Transform the data according to the number of pts from each sequence
        df_array = self.data.drop(columns=['SegIndex']).to_numpy()

        # if self.method == 'cyclic':
        self.n = int(df_array[:, 3].shape[0] / self.points)  # Number of oscillations/periods

        t_array = np.array([])
        temp_array = df_array[:, 3].reshape(int(self.n * 2), int(self.points / 2))
        for c in np.arange(0, temp_array.shape[0], 2):
            t_array = np.append(
                t_array,
                np.append(temp_array[c], temp_array[c + 1] + temp_array[c][-1]))
        self.t = t_array.reshape(int(self.n), int(self.points))

        self.half_n = int(self.t.shape[1] / 2)

        self.fn = df_array[:, 0].reshape(self.n, self.points)
        self.s = (self.fn / self.area) * 0.001  # N/m² => Pa / 1000 == 1 kPa
        self.h = df_array[:, 1].reshape(self.n, self.points)

        # elif self.method == 'total':
        t_total_array = np.array([])
        for c in np.arange(1, self.t.shape[0], 1):
            if c > 1:
                t_total_array = np.append(
                    t_total_array, self.t[c] + t_total_array[-1])
            else:
                t_total_array = np.append(
                    self.t[0], self.t[c] + self.t[0][-1])

        self.t_total = t_total_array
        self.fn_total = df_array[:, 0]
        self.s_total = (self.fn_total / self.area) * 0.001  # N/m² => Pa / 1000 == 1 kPa
        self.h_total = df_array[:, 1].min() - df_array[:, 1]

        # else:
        #     raise ValueError("'method' arg must be 'cyclic' to plot each sequence vs. strain or "
        #                      "'total' to plot the entire dynamic compression test")

        # print(df_array[:, 3])  # select values from 3rd column
        # max_id = self.data['Fn in N'].idxmax()  # index of max force value
        # max_h = self.data['h in mm'].iloc[max_id]  # find the height where the force is max

    def print_parameters(
            self,
            name,
            parameters
    ):
        s_pt = (self.t_total[-1] - self.t_total[0]) / len(self.t_total)

        if len(parameters) > 4:
            print(f'\n*** {name} FITTING PARAMETERS: ***\n\n'
                  f'- Amplitude: {abs(parameters[0]):.2f} N.\n'
                  f'- Damping Coef.: {parameters[1]:.2f}.\n'
                  f'- Angular frequency: {parameters[2]:.2f} rad/pts = {parameters[2] / s_pt:.1f} rad/s.\n'
                  f'- Frequency: {(parameters[2] / s_pt) / (2 * np.pi):.2f} Hz.\n'
                  f'- Phase: {parameters[3] * (180 / np.pi):.1f} degrees.\n'
                  f'- Displacement in y axis: {parameters[4]:.2f} N.\n')

        elif len(parameters) <= 4:
            print(f'\n*** {name} FITTING PARAMETERS: ***\n\n'
                  f'- Amplitude: {parameters[0]:.2f} N.\n'
                  f'- Damping Coef.: {parameters[1]:.2f}.\n'
                  f'- Angular frequency: {parameters[2]:.2f} rad/pts = {parameters[2] / s_pt:.1f} rad/s.\n'
                  f'- Frequency: {(parameters[2] / s_pt) / (2 * np.pi):.2f} Hz.\n')

    def linear_fit(
            self,
            slope_array,
            stdev_array,
            interactor
    ):
        # Convert time value to index
        self.i_index = np.where(self.t[interactor, :] < self.i_linreg)[-1][-1]
        self.f_index = np.where(self.t[interactor, :] < self.f_linreg)[-1][-1]
        optimal, covariance = curve_fit(
            linear_reg,
            self.t[interactor, self.i_index:self.f_index],
            self.s[interactor, self.i_index:self.f_index],
            p0=(2, 0)
        )
        error = np.sqrt(np.diag(covariance))

        slope_array = np.append(slope_array, optimal[0])
        stdev_array = np.append(stdev_array, error[0])

        return slope_array, stdev_array, optimal

    def sinusoid_fit(
            self,
            mode=str,
    ):
        if mode == 'normal':
            optimal_sin, covariance_sin = curve_fit(
                sinusoid,
                self.t_total,
                self.h_total,
                p0=(1.25, 2.9, 1.75, 1.2)
            )
            return optimal_sin, covariance_sin

        elif mode == 'damped':
            optimal_damped, covariance_damped = curve_fit(
                damped_sinusoid,
                self.t_total[45:],
                self.s_total[45:],
                p0=(1.5, 0.01, 2.9, 1.75, 1.2)
            )
            return optimal_damped, covariance_damped

        elif mode == 'absolute':
            optimal_abs, covariance_abs = curve_fit(
                abs_damped_sinusoid,
                self.t_total[45:],
                self.s_total[45:],
                p0=(3, 0.01, 1.75, 1.75)
            )
            return optimal_abs, covariance_abs

        else:
            return 'No fitting functions were selected.'

    def stress_peak(
            self,
            size
    ):  # Get the max stress values from a range and store the means and std dev
        x = np.array([])
        peak = np.array([])
        mean = np.array([])
        std = np.array([])
        maxs_index = np.argmax(self.fn, axis=1)

        for f in range(len(maxs_index)):
            x = np.append(x, self.t[f, maxs_index[f] - size:maxs_index[f] + size * 3])
            peak = np.append(peak, self.s[f, (maxs_index[f] - size):(maxs_index[f] + size * 3)])
            mean = np.append(mean, peak.mean())
            std = np.append(std, peak.std())

        x = x.reshape(self.n, int(x.shape[0] / self.n))
        peak = peak.reshape(self.n, x.shape[1])

        return x, mean, std, peak

    def stress_strain(
            self,
            color_series, color_linrange,  # Colors config
            plot_time=bool, plot_peak=bool, plot_fit=bool  # Additional plots
    ):
        ax = self.fig.add_subplot(self.gs[:, 0])
        ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(1)

        if plot_time:  # If True, it shows the Stress X Time plot
            ax.set_xlabel('Oscillation time (s)')
        else:
            ax.set_xlabel('Strain')
            ax.set_xticks([0, 0.5, 1, 1.5, 2])
            ax.set_xticklabels(['0%', '10%', '20%', '-10%', '-20%'])
            ax.xaxis.set_minor_locator(MultipleLocator(0.25))
        ax.set_xlim([0, 2])

        ax.set_ylabel('Stress (kPa)')
        ax.yaxis.set_major_locator(MultipleLocator(0.50))
        ax.yaxis.set_minor_locator(MultipleLocator(0.25))
        ax.set_ylim([-0.25, self.s.max() + self.s.max() * 0.15])


        # Plot the linear range and its values
        ax.axvspan(self.i_linreg, self.f_linreg, alpha=0.07, color=color_linrange)
        ax.text(self.i_linreg + 0.02, self.s.max() + self.s.max() * 0.1,
                f'Linear region', color=color_linrange, weight='bold')
        ax.text(self.i_linreg - 0.165, -0.1, f'{self.i_linreg_pct:.1f}%', color=color_linrange)
        ax.text(self.f_linreg + 0.020, -0.1, f'{self.f_linreg_pct:.1f}%', color=color_linrange)

        # Plot the peak range
        x_peak, peak_mean, peak_std, peak_val = self.stress_peak(self.peak_size)
        if plot_peak:
            ax.axvspan(x_peak[-1][0], x_peak[-1][-1], alpha=0.2, color='mediumpurple')
            ax.text(x_peak[-1][-1] + 0.03, self.s.max() + self.s.max() * 0.1,
                    f'Peak region', color='mediumpurple', weight='bold')

        for i in np.arange(0, self.n, 1):

            # Plot experimental data
            ax.scatter(
                np.append(self.t[i, :self.half_n], self.t[i, self.half_n:]), self.s[i, :],
                label=f'#{i + 1} period', color=color_series, edgecolors='none', s=30, alpha=(0.8 - 0.25 * i))

            # If plot_peak is True, plot peak data
            if plot_peak:
                ax.scatter(
                    x_peak[i, :], peak_val[i, :],
                    color=color_series, edgecolors='indigo', linewidths=0.75, s=30, alpha=0.70)

            # If plot_fit is True, plot fitted curve
            self.slope_val, self.slope_std, popt = self.linear_fit(self.slope_val, self.slope_std, i)
            if plot_fit:
                ax.plot(
                    self.t[i, self.i_index + 1:self.f_index + 1],
                    linear_reg(self.t[i, self.i_index + 1:self.f_index + 1], popt[0], popt[1]),
                    color=color_linrange, alpha=(0.75 - 0.12 * i), lw=1.3)
                ax.scatter(
                    [self.t[i, self.i_index + 1], self.t[i, self.f_index + 1]],
                    [self.s[i, self.i_index + 1], self.s[i, self.f_index + 1]],
                    color=color_series, edgecolors=color_linrange, linewidths=0.75, s=30, alpha=1)

        ax.legend(frameon=False)

    def peak_period(
            self,
            color_series
    ):
        # if self.stress:
        #     ax = self.fig.add_subplot(self.gs[0, 1])
        # elif not self.stress:
        #     ax = self.fig.add_subplot(self.gs[0])

        if self.stress and not self.ym:
            ax = self.fig.add_subplot(self.gs[0, 1])

        elif not self.stress and not self.ym:
            ax = self.fig.add_subplot(self.gs[0])

        elif not self.stress and self.ym:
            ax = self.fig.add_subplot(self.gs[1])

        else:
            ax = self.fig.add_subplot(self.gs[0, 1])

        x_peak, peak_mean, peak_std, peak_val = self.stress_peak(self.peak_size)

        period_array = np.arange(1, self.n + 1, 1)

        ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(1)
        ax.set_xlabel('Period')
        ax.set_xticks(period_array)
        ax.set_xlim([period_array[0] - 0.7, period_array[-1] + 0.7])

        ax.set_ylabel('Peak stress (kPa)')
        ax.set_ylim([peak_mean.min() - peak_mean.min() * 0.05, peak_mean.max() + peak_mean.max() * 0.05])

        ax.errorbar(
            period_array, peak_mean, yerr=peak_std, alpha=1,
            fmt='o', markersize=8, color=color_series, markeredgecolor='#383838', markeredgewidth=1,
            capsize=3, capthick=1, elinewidth=1, ecolor='#383838', linestyle='-')

        for val in period_array:
            # Plot the mean values and the std dev of each peak
            # X of text a lil bit from the left of the center period
            # Y of the test in the center (peak_mean[val - 1])
            # of upper error bar (peak_std[val - 1] + peak_std[val - 1] * 0.1)
            ax.text(val - 0.37,
                    peak_mean[val - 1] + peak_std[val - 1] + peak_std[val - 1] * 0.2,
                    f'{peak_mean[val - 1]:.2f} ± {peak_std[val - 1]:.2f} kPa',
                    color='#383838', bbox=dict(facecolor='w', alpha=1, edgecolor='w', pad=0))

        diff = 100 - (peak_mean[0] / peak_mean[-1]) * 100
        diff_s = ''
        if diff > 0:
            diff_s = 'Increased'
        elif diff < 0:
            diff_s = 'Decreased'
        ax.text(period_array[0] - 0.6, peak_mean.min() - peak_mean.min() * 0.045,
                f'{diff_s} in: {abs(diff):.1f}%', color='#383838')

    def ym_period(
            self,
            color_series
    ):
        if self.stress and not self.peak:
            ax = self.fig.add_subplot(self.gs[0, 1])

        elif not self.stress and not self.peak:
            ax = self.fig.add_subplot(self.gs[0])

        elif not self.stress and self.peak:
            ax = self.fig.add_subplot(self.gs[1])

        else:
            ax = self.fig.add_subplot(self.gs[1, 1])

        period_array = np.arange(1, self.n + 1, 1)

        for i in period_array - 1:
            self.slope_val, self.slope_std, popt = self.linear_fit(self.slope_val, self.slope_std, i)

        ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(1)
        ax.set_xlabel('Period')
        ax.set_xticks(period_array)
        ax.set_xlim([period_array[0] - 0.7, period_array[-1] + 0.7])

        ax.set_ylabel("Young's modulus (kPa)")
        ax.set_ylim([self.slope_val.min() - self.slope_val.min() * 0.03,
                     self.slope_val.max() + self.slope_val.max() * 0.03])

        ax.errorbar(
            period_array, self.slope_val[:3], yerr=self.slope_std[:3], alpha=1,
            fmt='o', markersize=8, color=color_series, markeredgecolor='#383838', markeredgewidth=1,
            capsize=3, capthick=1, elinewidth=1, ecolor='#383838', linestyle='-')

        for val in period_array:  # Show the mean values and the std dev of each peak
            # X of text a lil bit from the left of the center period
            # Y of the test in the center (peak_mean[val - 1])
            # of upper error bar (peak_std[val - 1] + peak_std[val - 1] * 0.1)
            ax.text(val - 0.37,
                    self.slope_val[val - 1] + self.slope_std[val - 1] + self.slope_std[val - 1] * 0.7,
                    f'{self.slope_val[val - 1]:.2f} ± {self.slope_std[val - 1]:.2f} kPa',
                    color='#383838', bbox=dict(facecolor='w', alpha=1, edgecolor='w', pad=0))

        diff = 100 - (self.slope_val[0] / self.slope_val[-1]) * 100
        diff_s = ''
        if diff > 0:
            diff_s = 'Increased'
        elif diff < 0:
            diff_s = 'Decreased'
        ax.text(period_array[0] - 0.6, self.slope_val.min() - self.slope_val.min() * 0.025,
                f'{diff_s} in: {abs(diff):.1f}%', color='#383838')

    def cyclic_plot(
            self,
            peak_size=3,
            initial_strain=10,
            final_strain=18,
            stress=True,
            peak=True,
            ym=True,
            ratios=(3, 2),  # Charts aspect ratio
            plot_time=False, plot_peak=False, plot_fit=False,  # Additional plots
            color_series='dodgerblue', color_linrange='crimson'  # Colors from series and linear region, respectively
    ):
        self.peak_size = peak_size

        self.stress = stress
        self.peak = peak
        self.ym = ym

        self.i_linreg_pct = initial_strain  # Initial strain value for linear region
        self.f_linreg_pct = final_strain  # Final strain value for linear region
        self.i_linreg = (0.5 / 10) * self.i_linreg_pct  # Convert the strain values to time values
        self.f_linreg = (0.5 / 10) * self.f_linreg_pct  # (0.5 s)/(10 %) × x%

        self.fig.suptitle(f'Dynamic compression - Cycle analysis ({self.data_path})', alpha=0.9)

        if self.stress and self.peak and self.ym:
            self.gs = GridSpec(2, 2, width_ratios=ratios)

            self.stress_strain(color_series, color_linrange, plot_time, plot_peak, plot_fit)
            self.peak_period(color_series)
            self.ym_period(color_series)

        elif self.peak and self.ym and not self.stress:
            self.gs = GridSpec(2, 1)

            self.peak_period(color_series)
            self.ym_period(color_series)

        elif self.stress and self.peak and not self.ym:
            self.gs = GridSpec(1, 2, width_ratios=ratios)

            self.stress_strain(color_series, color_linrange, plot_time, plot_peak, plot_fit)
            self.peak_period(color_series)

        elif self.stress and self.ym and not self.peak:
            self.gs = GridSpec(1, 2, width_ratios=ratios)

            self.stress_strain(color_series, color_linrange, plot_time, plot_peak, plot_fit)
            self.ym_period(color_series)

        else:
            self.gs = GridSpec(1, 1)

            if self.stress and not self.ym and not self.peak:
                self.stress_strain(color_series, color_linrange, plot_time, plot_peak, plot_fit)

            if not self.stress and self.peak and not self.ym:
                self.peak_period(color_series)

            if not self.stress and not self.peak and self.ym:
                self.ym_period(color_series)

        return self.fig

    def total_plot(
            self,
            normal=False, damped=False, absolute=False,  # Fitting plots
            plot_exp_h=True, plot_time=False,  # Additional plots
            colorax1='dodgerblue', colorax2='silver'  # Colors from stress and height, respectively
    ):
        self.plot_exp_h = plot_exp_h

        popt_sin, _ = self.sinusoid_fit('normal')
        popt_dpd, _ = self.sinusoid_fit('damped')
        popt_abs, _ = self.sinusoid_fit('absolute')

        # Plots configs
        self.gs = GridSpec(1, 1)
        ax1 = self.fig.add_subplot(self.gs[:, 0])
        self.fig.suptitle(f'Dynamic compression - Full oscillation ({self.data_path})', alpha=0.9)

        # Left axis configs
        ax1.set_xlim([0, 2 * self.n])
        if plot_time:  # If True, it shows the Stress X Time plot
            ax1.set_xlabel('Time (s)')
        else:
            ax1.set_xlabel('Strain')
            ax1.set_xticks(np.arange(0, 2 * self.n + 0.5, 0.5))
            ax1.set_xticklabels(np.append(np.array(['0%', '10%', '20%', '-10%'] * self.n), ['0%']))
        ax1.set_ylabel('Stress (kPa)')
        ax1.yaxis.set_major_locator(MultipleLocator(0.50))
        ax1.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)
        # ax1.yaxis.set_minor_locator(MultipleLocator(0.25))

        # Experimental data
        ax1.scatter(
            self.t_total,
            self.s_total,
            color=colorax1, alpha=0.5, s=35, marker='o', edgecolors='none')

        # Fitted curves
        if damped:
            ax1.plot(
                self.t_total[45:],
                damped_sinusoid(self.t_total[45:], popt_dpd[0], popt_dpd[1], popt_dpd[2], popt_dpd[3],
                                popt_dpd[4]),
                color=colorax1, alpha=0.75, label=f'Damped sinusoid - Damping coef.: {popt_dpd[1]:.2f}')
            self.print_parameters('Damped sinusoid', popt_dpd)

        if absolute:
            ax1.plot(
                self.t_total[45:],
                abs_damped_sinusoid(self.t_total[45:], popt_abs[0], popt_abs[1], popt_abs[2], popt_abs[3]),
                color=colorax1, alpha=0.75, label=f'Abs damped sinusoid - Damping coef.: {popt_abs[1]:.2f}', ls='--')
            self.print_parameters('Abs damped sinusoid', popt_abs)

        # Right axis configs
        if self.plot_exp_h:
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            ax2.spines['left'].set_color(colorax1)
            ax2.spines['right'].set_color(colorax2)
            ax2.set_xlim([0, 2 * self.n])

            ax2.set_ylabel('Height (mm)', color=colorax2)
            ax2.tick_params(axis='y', labelcolor=colorax2, colors=colorax2)
            ax2.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)
            ax2.set_ylim([self.h_total.max(),
                          self.h_total.min() + self.h_total.min() * 0.1])
            ax2.yaxis.set_major_locator(MultipleLocator(0.5))

            ax1.set_ylabel('Stress (kPa)', color=colorax1)
            ax1.tick_params(axis='y', labelcolor=colorax1, colors=colorax1)

            # Experimental data
            ax2.scatter(
                self.t_total,
                self.h_total,
                color=colorax2, alpha=0.25, s=25, marker='o', edgecolors='none')

        # Fitted curve
        if normal:
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            ax2.spines['left'].set_color(colorax1)
            ax2.spines['right'].set_color(colorax2)
            ax2.set_xlim([0, 2 * self.n])

            ax2.set_ylabel('Height (mm)', color=colorax2)
            ax2.tick_params(axis='y', labelcolor=colorax2, colors=colorax2)
            ax2.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)
            ax2.set_ylim([self.h_total.max(),
                          self.h_total.min() + self.h_total.min() * 0.1])
            ax2.yaxis.set_major_locator(MultipleLocator(0.5))

            ax1.set_ylabel('Stress (kPa)', color=colorax1)
            ax1.tick_params(axis='y', labelcolor=colorax1, colors=colorax1)

            ax2.plot(
                self.t_total,
                sinusoid(self.t_total, popt_sin[0], popt_sin[1], popt_sin[2], popt_sin[3]),
                color=colorax2, alpha=0.75, label=f'Fitted damped sinusoid')

        self.fig.tight_layout()  # otherwise the right y-label is slightly clipped
        ax1.legend(loc=1, ncol=2, frameon=False)
        # ax1.text(13.7 * cm, 7.7 * cm, 'Damping coef.: -0.02', )  # Show the damping coef in chart]

        return self.fig


# Global configs
np.set_printoptions(threshold=np.inf)  # print the entire array
cm = 1 / 2.54  # centimeters in inches
fonts(folder_path='C:/Users/petrus.kirsten/AppData/Local/Microsoft/Windows/Fonts/')

if __name__ == "__main__":
    path = "C:/Users/petrus.kirsten/PycharmProjects/RheometerPlots/data/haribo-9v2.csv"

    data = DynamicCompression(
        path,
        196)
    DynamicCompression.cyclic_plot(
        data,
        peak_size=5,
        stress=True, peak=True, ym=True,
        plot_peak=True, plot_fit=True)
    plt.show()

    # data = Plotting(
    #     data,
    #     n_points)
    # Plotting.total_plot(
    #     data,
    #     True, True, True,
    #     True)
    # plt.show()

    print('DynamicCompression exec as main.')
