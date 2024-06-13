import os
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
    def __init__(
            self,
            data_path,
            cycles,
            mode,
            figure_size=(34, 15)
    ):
        self.data_path = data_path
        self.nCycles = cycles
        self.figure_size = figure_size

        # Figure vars
        self.gs = None
        self.fig = plt.figure(figsize=(self.figure_size[0] * cm, self.figure_size[1] * cm))
        self.fig.subplots_adjust(hspace=0)
        # Plot vars
        self.area = np.pi * 0.015 ** 2  # 30 mm diamater circle => r = 0.015 m => S = 0.0007 m²
        self.peakSize = 3
        self.plotExpHeight, self.plotStress, self.plotPeak, self.plotYoung = None, None, None, None
        self.i_linreg_pct = 7.5  # Initial strain value for linear region
        self.f_linreg_pct = 18  # Final strain value for linear region
        self.i_linreg = (0.5 / 10) * self.i_linreg_pct  # Convert the strain values to time values
        self.f_linreg = (0.5 / 10) * self.f_linreg_pct  # (0.5 s)/(10 %) × x%
        self.i_index, self.f_index = 0, 0
        # Linear fitting vars
        self.slope_val, self.slope_std = np.array([]), np.array([])
        # Data vars
        self.data = pd.read_csv(self.data_path[0])
        self.timeData, self.heightData, self.forceData, self.stressData = self.getData(mode)

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
            colorSeries='dodgerblue', colorLinRange='crimson'  # Colors from series and linear region, respectively
    ):
        self.peakSize = peak_size

        self.plotStress = stress
        self.plotPeak = peak
        self.plotYoung = ym

        self.i_linreg_pct = initial_strain  # Initial strain value for linear region
        self.f_linreg_pct = final_strain  # Final strain value for linear region
        self.i_linreg = (0.5 / 10) * self.i_linreg_pct  # Convert the strain values to time values
        self.f_linreg = (0.5 / 10) * self.f_linreg_pct  # (0.5 s)/(10 %) × x%

        self.fig.suptitle(f'Dynamic compression - Cycle analysis '
                          f'({os.path.basename(self.data_path[0]).split("/")[-1]})', alpha=0.9)

        if self.plotStress and self.plotPeak and self.plotYoung:
            self.gs = GridSpec(2, 2, width_ratios=ratios)

            self.stress_strain(colorSeries, colorLinRange, plot_time, plot_peak, plot_fit)
            self.peak_period(colorSeries)
            self.ym_period(colorSeries)

        elif self.plotPeak and self.plotYoung and not self.plotStress:
            self.gs = GridSpec(2, 1)

            self.peak_period(colorSeries)
            self.ym_period(colorSeries)

        elif self.plotStress and self.plotPeak and not self.plotYoung:
            self.gs = GridSpec(1, 2, width_ratios=ratios)

            self.stress_strain(colorSeries, colorLinRange, plot_time, plot_peak, plot_fit)
            self.peak_period(colorSeries)

        elif self.plotStress and self.plotYoung and not self.plotPeak:
            self.gs = GridSpec(1, 2, width_ratios=ratios)

            self.stress_strain(colorSeries, colorLinRange, plot_time, plot_peak, plot_fit)
            self.ym_period(colorSeries)

        else:
            self.gs = GridSpec(1, 1)

            if self.plotStress and not self.plotYoung and not self.plotPeak:
                self.stress_strain(colorSeries, colorLinRange, plot_time, plot_peak, plot_fit)

            if not self.plotStress and self.plotPeak and not self.plotYoung:
                self.peak_period(colorSeries)

            if not self.plotStress and not self.plotPeak and self.plotYoung:
                self.ym_period(colorSeries)

        return self.fig

    def total_plot(
            self,
            normal=False, damped=False, absolute=False,  # Fitting plots
            plot_exp_h=True,  # Additional plot
            colorax1='dodgerblue', colorax2='silver'  # Colors from stress and height, respectively
    ):
        self.plotExpHeight = plot_exp_h

        popt_sin, _ = self.sinusoid_fit('normal')
        popt_dpd, _ = self.sinusoid_fit('damped')
        popt_abs, _ = self.sinusoid_fit('absolute')

        # Plots configs
        self.gs = GridSpec(1, 1)
        ax1 = self.fig.add_subplot(self.gs[:, 0])
        self.fig.suptitle(f'Dynamic compression - Full oscillation '
                          f'({os.path.basename(self.data_path[0]).split("/")[-1]})', alpha=0.9)

        # Left axis configs
        ax1.set_xlim([0, 2 * self.nCycles])
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Stress (kPa)')
        # ax1.yaxis.set_major_locator(MultipleLocator(0.50))
        ax1.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)
        # ax1.yaxis.set_minor_locator(MultipleLocator(0.25))

        # Experimental data
        ax1.scatter(
            self.timeData,
            self.stressData,
            color=colorax1, alpha=0.5, s=35, marker='o', edgecolors='none')

        # Fitted curves
        if damped:
            ax1.plot(
                self.timeData[45:],
                damped_sinusoid(self.timeData[45:], popt_dpd[0], popt_dpd[1], popt_dpd[2], popt_dpd[3],
                                popt_dpd[4]),
                color=colorax1, alpha=0.75, label=f'Damped sinusoid - Damping coef.: {popt_dpd[1]:.2f}')
            self.printParameters('Damped sinusoid', popt_dpd)

        if absolute:
            ax1.plot(
                self.timeData[45:],
                abs_damped_sinusoid(self.timeData[45:], popt_abs[0], popt_abs[1], popt_abs[2], popt_abs[3]),
                color=colorax1, alpha=0.75, label=f'Abs damped sinusoid - Damping coef.: {popt_abs[1]:.2f}', ls='--')
            self.printParameters('Abs damped sinusoid', popt_abs)

        # Right axis configs
        if self.plotExpHeight:
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            ax2.spines['left'].set_color(colorax1)
            ax2.spines['right'].set_color(colorax2)
            ax2.set_xlim([0, 2 * self.nCycles])

            ax2.set_ylabel('Height (mm)', color=colorax2)
            ax2.tick_params(axis='y', labelcolor=colorax2, colors=colorax2)
            ax2.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)
            ax2.set_ylim([self.heightData.max(),
                          self.heightData.min() + self.heightData.min() * 0.1])
            ax2.yaxis.set_major_locator(MultipleLocator(0.5))

            ax1.set_ylabel('Stress (kPa)', color=colorax1)
            ax1.tick_params(axis='y', labelcolor=colorax1, colors=colorax1)

            # Experimental data
            ax2.scatter(
                self.timeData,
                self.heightData,
                color=colorax2, alpha=0.25, s=25, marker='o', edgecolors='none')

        # Fitted curve
        if normal:
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            ax2.spines['left'].set_color(colorax1)
            ax2.spines['right'].set_color(colorax2)
            ax2.set_xlim([0, 2 * self.nCycles])

            ax2.set_ylabel('Height (mm)', color=colorax2)
            ax2.tick_params(axis='y', labelcolor=colorax2, colors=colorax2)
            ax2.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)
            ax2.set_ylim([self.heightData.max(),
                          self.heightData.min() + self.heightData.min() * 0.1])
            ax2.yaxis.set_major_locator(MultipleLocator(0.5))

            ax1.set_ylabel('Stress (kPa)', color=colorax1)
            ax1.tick_params(axis='y', labelcolor=colorax1, colors=colorax1)

            ax2.plot(
                self.timeData,
                sinusoid(self.timeData, popt_sin[0], popt_sin[1], popt_sin[2], popt_sin[3]),
                color=colorax2, alpha=0.75, label=f'Fitted damped sinusoid')

        self.fig.tight_layout()  # otherwise the right y-label is slightly clipped
        ax1.legend(loc=1, ncol=2, frameon=False)
        # ax1.text(13.7 * cm, 7.7 * cm, 'Damping coef.: -0.02', )  # Show the damping coef in chart]

        return self.fig

    def linear_fit(
            self,
            slope_array,
            stdev_array,
            interactor
    ):
        # Convert time value to index
        self.i_index = np.where(self.timeData[interactor, :] < self.i_linreg)[-1][-1]
        self.f_index = np.where(self.timeData[interactor, :] < self.f_linreg)[-1][-1]
        optimal, covariance = curve_fit(
            linear_reg,
            self.timeData[interactor, self.i_index:self.f_index],
            self.stressData[interactor, self.i_index:self.f_index],
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
                self.timeData,
                self.heightData,
                p0=(1.25, 2.9, 1.75, 1.2)
            )
            return optimal_sin, covariance_sin

        elif mode == 'damped':
            optimal_damped, covariance_damped = curve_fit(
                damped_sinusoid,
                self.timeData[45:],
                self.stressData[45:],
                p0=(1.5, 0.01, 2.9, 1.75, 1.2)
            )
            return optimal_damped, covariance_damped

        elif mode == 'absolute':
            optimal_abs, covariance_abs = curve_fit(
                abs_damped_sinusoid,
                self.timeData[45:],
                self.stressData[45:],
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
        maxs_index = np.argmax(self.forceData, axis=1)

        for f in range(len(maxs_index)):
            x = np.append(x, self.timeData[f, maxs_index[f] - size:maxs_index[f] + size * 3])
            peak = np.append(peak, self.stressData[f, (maxs_index[f] - size):(maxs_index[f] + size * 3)])
            mean = np.append(mean, peak.mean())
            std = np.append(std, peak.std())

        x = x.reshape(self.nCycles, int(x.shape[0] / self.nCycles))
        peak = peak.reshape(self.nCycles, x.shape[1])

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
        ax.set_ylim([-0.25, self.stressData.max() + self.stressData.max() * 0.15])

        # Plot the linear range and its values
        ax.axvspan(self.i_linreg, self.f_linreg, alpha=0.07, color=color_linrange)
        ax.text(self.i_linreg + 0.02, self.stressData.max() + self.stressData.max() * 0.1,
                f'Linear region', color=color_linrange, weight='bold')
        ax.text(self.i_linreg - 0.165, -0.1, f'{self.i_linreg_pct:.1f}%', color=color_linrange)
        ax.text(self.f_linreg + 0.020, -0.1, f'{self.f_linreg_pct:.1f}%', color=color_linrange)

        # Plot the peak range
        x_peak, peak_mean, peak_std, peak_val = self.stress_peak(self.peakSize)
        if plot_peak:
            ax.axvspan(x_peak[-1][0], x_peak[-1][-1], alpha=0.2, color='mediumpurple')
            ax.text(x_peak[-1][-1] + 0.03, self.stressData.max() + self.stressData.max() * 0.1,
                    f'Peak region', color='mediumpurple', weight='bold')

        for i in np.arange(0, self.nCycles, 1):

            # Plot experimental data
            ax.scatter(
                np.append(self.timeData[i, :self.timeData.shape[1] // 2], self.timeData[i, self.timeData.shape[1] // 2:]), self.stressData[i, :],
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
                    self.timeData[i, self.i_index + 1:self.f_index + 1],
                    linear_reg(self.timeData[i, self.i_index + 1:self.f_index + 1], popt[0], popt[1]),
                    color=color_linrange, alpha=(0.75 - 0.12 * i), lw=1.3)
                ax.scatter(
                    [self.timeData[i, self.i_index + 1], self.timeData[i, self.f_index + 1]],
                    [self.stressData[i, self.i_index + 1], self.stressData[i, self.f_index + 1]],
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

        if self.stressData and not self.plotYoung:
            ax = self.fig.add_subplot(self.gs[0, 1])

        elif not self.stressData and not self.plotYoung:
            ax = self.fig.add_subplot(self.gs[0])

        elif not self.stressData and self.plotYoung:
            ax = self.fig.add_subplot(self.gs[1])

        else:
            ax = self.fig.add_subplot(self.gs[0, 1])

        x_peak, peak_mean, peak_std, peak_val = self.stress_peak(self.peakSize)

        period_array = np.arange(1, self.nCycles + 1, 1)

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
        if self.stressData and not self.plotPeak:
            ax = self.fig.add_subplot(self.gs[0, 1])

        elif not self.stressData and not self.plotPeak:
            ax = self.fig.add_subplot(self.gs[0])

        elif not self.stressData and self.plotPeak:
            ax = self.fig.add_subplot(self.gs[1])

        else:
            ax = self.fig.add_subplot(self.gs[1, 1])

        period_array = np.arange(1, self.nCycles + 1, 1)

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

    def printParameters(
            self,
            name,
            parameters
    ):
        s_pt = (self.timeData[-1] - self.timeData[0]) / len(self.timeData)

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

    def getData(
            self, mode
    ):
        # TODO: arrumar a saída dos dados de tempo. Precisa 
        t_seg = self.data['t_seg in s'].to_numpy()

        tempTime = np.array([])
        t_segShaped = t_seg.reshape(2 * self.nCycles, len(t_seg) // (2 * self.nCycles))

        for c in np.arange(0, t_segShaped.shape[0], 2):
            tempTime = np.append(
                tempTime,
                np.append(t_segShaped[c], t_segShaped[c + 1] + t_segShaped[c][-1]))
        tempTime = tempTime.reshape(t_segShaped.shape[0] // 2, t_segShaped.shape[1] * 2)

        tempTotalTime = np.array([])
        for c in np.arange(1, tempTime.shape[0], 1):
            if c > 1:
                tempTotalTime = np.append(
                    tempTotalTime, tempTime[c] + tempTotalTime[-1])
            else:
                tempTotalTime = np.append(
                    tempTime[0], tempTime[c] + tempTime[0][-1])

        height = self.data['h in mm'].to_numpy()
        force = self.data['Fn in N'].to_numpy()
        stress = (force / self.area) * 0.001  # N/m² => Pa / 1000 == 1 kPa

        if mode == 'Total':
            return tempTotalTime, height, force, stress

        if mode == 'Cyclic':
            height = height.reshape(t_segShaped.shape[0] // 2, t_segShaped.shape[1] * 2)
            height = force.reshape(t_segShaped.shape[0] // 2, t_segShaped.shape[1] * 2)
            height = stress.reshape(t_segShaped.shape[0] // 2, t_segShaped.shape[1] * 2)

            return tempTime, height, force, stress


class Sweep:
    def __init__(
            self,
            data_path,
            figure_size=(22, 15)
    ):
        self.data_path = data_path
        self.figure_size = figure_size

        self.fig = plt.figure(figsize=(self.figure_size[0] * cm, self.figure_size[1] * cm))
        self.gs = GridSpec(1, 1)
        self.fig.subplots_adjust(hspace=0)

        # Collecting the data
        self.data = None
        self.timeTotal = None
        self.timeElement = None
        self.strainStress = None
        self.compViscosity = None
        self.temperature = None
        self.storageModulus = None
        self.storageModulusErr = None
        self.lossModulus = None
        self.lossModulusErr = None
        self.shearStress = None
        self.frequency = None
        self.angVeloc = None

    def stress(
            self,
            mode='Shear Stress',
            colorStorage='dodgerblue', colorLoss='hotpink'
    ):
        self.getData(mode)
        ax1, ax2 = self.configPlot(mode)

        ax1.scatter(
            self.shearStress, self.storageModulus,
            color=colorStorage, alpha=0.75, s=45, marker='o', edgecolors=colorStorage)

        ax2.scatter(
            self.shearStress, self.lossModulus,
            color=colorLoss, alpha=0.75, s=45, marker='o', edgecolors=colorLoss)

        self.fig.tight_layout()  # otherwise the right y-label is slightly clipped

    def oscilatory(
            self,
            mode='Freq',
            colorStorage='dodgerblue', colorLoss='hotpink'
    ):
        self.getData(mode)
        ax1, ax2 = self.configPlot(mode)

        ax1.errorbar(
            self.angVeloc, self.storageModulus, yerr=self.storageModulusErr,
            color=colorStorage, alpha=0.75, markersize=7, fmt='o',
            markeredgecolor=colorStorage, markeredgewidth=1,
            capsize=3, capthick=1, elinewidth=1, ecolor=colorStorage)

        ax2.errorbar(
            self.angVeloc, self.lossModulus, yerr=self.lossModulusErr,
            color=colorLoss, alpha=0.75, markersize=7, fmt='o',
            markeredgecolor=colorLoss, markeredgewidth=1,
            capsize=3, capthick=1, elinewidth=1, ecolor=colorLoss)

        self.fig.tight_layout()  # otherwise the right y-label is slightly clipped

    def getData(
            self, xAxis
    ):
        gPrime, gDouble = np.array([]), np.array([])

        for file in range(len(self.data_path)):
            self.data = pd.read_csv(self.data_path[file])

            self.timeTotal = self.data['t in s'].to_numpy()
            self.timeElement = self.data['t_seg in s'].to_numpy()
            self.strainStress = self.data['ɣ in %'].to_numpy()
            self.compViscosity = self.data['|η*| in mPas'].to_numpy()
            self.temperature = self.data['T in °C'].to_numpy()

            if xAxis == 'Shear Stress':
                self.shearStress = self.data['τ in Pa'].to_numpy()
            if xAxis == 'Freq':
                self.frequency = self.data['f in Hz'].to_numpy()
                self.angVeloc = 2 * np.pi * self.frequency

            gPrime = np.append(gPrime, self.data["G' in Pa"].to_numpy())
            gDouble = np.append(gDouble, self.data['G" in Pa'].to_numpy())

        gPrime = gPrime.reshape(
            len(self.data_path), int(len(gPrime) / len(self.data_path)))
        gDouble = gDouble.reshape(
            len(self.data_path), int(len(gDouble) / len(self.data_path)))

        self.storageModulus = gPrime.mean(axis=0)
        self.storageModulusErr = gPrime.std(axis=0)
        self.lossModulus = gDouble.mean(axis=0)
        self.lossModulusErr = gDouble.std(axis=0)

        return

    def configPlot(
            self,
            mode,
            colorStorage='dodgerblue', colorLoss='hotpink'
    ):
        plt.style.use('seaborn-v0_8-ticks')
        ax1 = self.fig.add_subplot(self.gs[:, 0])

        # Right axis configs
        ax2 = ax1.twinx()
        ax2.set_yscale('log')
        ax2.set_xscale('log')
        ax2.set_ylabel('Loss modulus G" (Pa)', color=colorLoss)
        ax2.set_ylim(
            [int(round(np.min(self.lossModulus) - np.min(self.lossModulus) * 0.7, -2)),
             int(round(np.max(self.lossModulus) + np.max(self.lossModulus) * 2.7, -3))])
        ax2.tick_params(axis='y', which='major', labelcolor=colorLoss, colors=colorLoss)
        ax2.tick_params(axis='y', which='minor', labelcolor=colorLoss, colors=colorLoss, labelsize=8)
        ax2.spines['right'].set_color(colorLoss)
        ax2.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)

        # Left axis configs
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.set_ylabel("Storage modulus G' (Pa)", color=colorStorage)
        ax1.set_ylim(
            [int(round(np.min(self.storageModulus) - np.min(self.storageModulus) * 0.8, -3)),
             int(round(np.max(self.storageModulus) + np.max(self.storageModulus) * 0.4, -4))])
        ax1.tick_params(axis='y', which='major', labelcolor=colorStorage, colors=colorStorage)
        ax1.tick_params(axis='y', which='minor', labelcolor=colorStorage, colors=colorStorage, labelsize=8)
        ax2.spines['left'].set_color(colorStorage)
        ax1.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)

        if mode == 'Freq':
            self.fig.suptitle(f'Frequency sweeps '
                              f'({os.path.basename(self.data_path[0]).split("/")[-1]})', alpha=0.9)
            ax1.set_xlabel('Angular velocity (rad/s)')
            ax1.set_xlim([self.angVeloc[0], round(self.angVeloc[-1], -1)])
            ax2.set_xlim([self.angVeloc[0], round(self.angVeloc[-1], -1)])

        if mode == 'Shear Stress':
            self.fig.suptitle(f'Stress sweeps '
                              f'({os.path.basename(self.data_path[0]).split("/")[-1]})', alpha=0.9)
            ax1.set_xlabel('Shear stress (Pa)')
            ax1.set_xlim([self.shearStress[0], round(self.shearStress[-1], -1)])
            ax2.set_xlim([self.shearStress[0], round(self.shearStress[-1], -1)])

        return ax1, ax2


# Global configs
np.set_printoptions(threshold=np.inf)  # print the entire array
cm = 1 / 2.54  # centimeters in inches
fonts(folder_path='C:/Users/petrus.kirsten/AppData/Local/Microsoft/Windows/Fonts/')

if __name__ == "__main__":
    print('plotting.py exec as main.')

    # data = DynamicCompression(
    #     path,
    #     196)
    # DynamicCompression.cyclic_plot(
    #     data,
    #     peak_size=5,
    #     stress=True, peak=True, ym=True,
    #     plot_peak=True, plot_fit=True)
    # plt.show()

    # data = rheoplots(
    #     data,
    #     n_points)
    # rheoplots.total_plot(
    #     data,
    #     True, True, True,
    #     True)
    # plt.show()

    path = "C:/Users/petrus.kirsten/PycharmProjects/RheometerPlots/data/frequency-sweep_BB_280524-1.csv"

    data = Sweep(data_path=path)

    Sweep.oscilatory(data)

    plt.show()
