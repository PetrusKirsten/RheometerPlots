import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from matplotlib.font_manager import FontProperties


def fonts(folder_path, small=10, medium=12):  # To config different fonts but it isn't working with these
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


def meanCteRange(array, threshold=200):
    ranges = []
    init_range = 0

    for i in range(1, len(array)):
        if abs(array[i] - array[i - 1]) > threshold:
            if init_range < i - 1:
                ranges.append((init_range, i - 1))
            init_range = i

    if init_range < len(array) - 1:
        ranges.append((init_range, len(array) - 1))

    end_linRange = np.array(ranges)[0][-1]

    return [array[:end_linRange].mean(), end_linRange]


class DynamicCompression:
    def __init__(
            self,
            data_path,
            cycles,
            mode,
            figure_size=(34, 14),
            dpi=300
    ):
        self.data_path = data_path
        self.nCycles = cycles
        self.fileTitle = os.path.basename(self.data_path[0]).split("/")[-1].split(".")[0]
        self.fileFolder = os.path.dirname(self.data_path[0])
        self.filePath = self.fileFolder + '\\' + self.fileTitle
        self.figSize = figure_size
        self.dpi = dpi  # TODO: ajustar a config do DPI

        # Figure vars
        self.fig = plt.figure(self.fileTitle, figsize=(self.figSize[0] * cm, self.figSize[1] * cm))
        self.fig.subplots_adjust(hspace=0)
        self.gs = None
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
        self.data = pd.read_excel(self.data_path[0])
        self.timeData, self.heightData, self.forceData, self.stressData = self.getData(mode)

    def getData(
            self, mode
    ):
        t_seg = self.data['t_seg in s'].to_numpy()

        tempTimeCyclic = np.array([])
        t_segShaped = t_seg.reshape(2 * self.nCycles, len(t_seg) // (2 * self.nCycles))

        for c in np.arange(0, t_segShaped.shape[0], 2):
            tempTimeCyclic = np.append(
                tempTimeCyclic,
                np.append(t_segShaped[c], t_segShaped[c + 1] + t_segShaped[c][-1]))
        tempTimeCyclic = tempTimeCyclic.reshape(t_segShaped.shape[0] // 2, t_segShaped.shape[1] * 2)

        tempTime = np.array([])
        for c in np.arange(1, tempTimeCyclic.shape[0], 1):
            if c > 1:
                tempTime = np.append(
                    tempTime, tempTimeCyclic[c] + tempTime[-1])
            else:
                tempTime = np.append(
                    tempTimeCyclic[0], tempTimeCyclic[c] + tempTimeCyclic[0][-1])

        height = self.data['h in mm'].to_numpy()
        height = abs(height * 100 / height.max() - 100)
        # height = height - height.min()
        # height = 6.5 * height / height.max()
        force = self.data['Fn in N'].to_numpy()
        stress = (force / self.area) * 0.001  # N/m² => Pa / 1000 == 1 kPa

        if mode == 'Total':
            return tempTime, height, force, stress

        if mode == 'Cyclic':
            heightCyclic = height.reshape(t_segShaped.shape[0] // 2, t_segShaped.shape[1] * 2)
            forceCyclic = force.reshape(t_segShaped.shape[0] // 2, t_segShaped.shape[1] * 2)
            stressCyclic = stress.reshape(t_segShaped.shape[0] // 2, t_segShaped.shape[1] * 2)

            return tempTimeCyclic, heightCyclic, forceCyclic, stressCyclic

    def plotTotal(
            self,
            normal=False, damped=False, absolute=False,  # Fitting plots
            plot_exp_h=True,  # Additional plot
            colorax1='dodgerblue', colorax2='silver'  # Colors from stress and height, respectively
    ):
        self.plotExpHeight = plot_exp_h

        # Fitting
        popt_sin, _ = self.fitSinusoid('normal')
        popt_dpd, _ = self.fitSinusoid('damped')
        popt_abs, _ = self.fitSinusoid('absolute')

        # Plots configs
        plt.style.use('seaborn-v0_8-ticks')
        self.gs = GridSpec(1, 1)
        ax1 = self.fig.add_subplot(self.gs[:, 0])
        self.fig.suptitle(f'Dynamic compression - Full oscillation '
                          f'({self.fileTitle})', alpha=0.9)

        # Left axis configs
        ax1.set_xlabel('Time (s)')
        ax1.set_xlim([0, 2 * self.nCycles])
        ax1.set_ylabel('Stress (kPa)')
        ax1.set_ylim([self.stressData.min() - self.stressData.min() * 0.1,
                      self.stressData.max() + self.stressData.max() * 0.1])
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

            ax2.set_ylabel('Strain (%)', color=colorax2)
            ax2.tick_params(axis='y', labelcolor=colorax2, colors=colorax2)
            ax2.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)
            ax2.set_ylim([self.heightData.min() - self.heightData.min() * 0.1,
                          self.heightData.max() + self.heightData.max() * 0.1])
            # ax2.yaxis.set_major_locator(MultipleLocator(0.5))

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

            ax2.set_ylabel('Strain (%)', color=colorax2)
            ax2.tick_params(axis='y', labelcolor=colorax2, colors=colorax2)
            ax2.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)
            ax2.set_ylim([self.heightData.min() - self.heightData.min() * 0.1,
                          self.heightData.max() + self.heightData.max() * 0.1])
            # ax2.yaxis.set_major_locator(MultipleLocator(0.5))

            ax1.set_ylabel('Stress (kPa)', color=colorax1)
            ax1.tick_params(axis='y', labelcolor=colorax1, colors=colorax1)

            ax2.plot(
                self.timeData,
                sinusoid(self.timeData, popt_sin[0], popt_sin[1], popt_sin[2], popt_sin[3]),
                color=colorax2, alpha=0.75, label=f'Fitted damped sinusoid')

        self.fig.tight_layout()  # otherwise the right y-label is slightly clipped
        ax1.legend(loc=1, ncol=2, frameon=False)
        # ax1.text(13.7 * cm, 7.7 * cm, 'Damping coef.: -0.02', )  # Show the damping coef in chart]
        self.fig.savefig(f'{self.filePath}.png', dpi=self.dpi)
        print(f'Chart saved at {self.fileFolder}')

        return self.fig

    def plotCyclic(
            self,
            peak_size=3,
            initial_strain=10,
            final_strain=18,
            stress=True,
            peak=True,
            ym=True,
            ratios=(3, 2),  # Charts aspect ratio
            plotPeak=False, plotFit=False,  # Additional plots
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

        plt.style.use('seaborn-v0_8-ticks')
        self.fig.suptitle(f'Dynamic compression - Cycle analysis '
                          f'({self.fileTitle})', alpha=0.9)

        if self.plotStress and self.plotPeak and self.plotYoung:
            self.gs = GridSpec(2, 2, width_ratios=ratios)

            self.cyclicStress(colorSeries, colorLinRange, plotPeak, plotFit)
            self.cyclicPeak(colorSeries)
            self.cyclicYoung(colorSeries)

        elif self.plotPeak and self.plotYoung and not self.plotStress:
            self.gs = GridSpec(2, 1)

            self.cyclicPeak(colorSeries)
            self.cyclicYoung(colorSeries)

        elif self.plotStress and self.plotPeak and not self.plotYoung:
            self.gs = GridSpec(1, 2, width_ratios=ratios)

            self.cyclicStress(colorSeries, colorLinRange, plotPeak, plotFit)
            self.cyclicPeak(colorSeries)

        elif self.plotStress and self.plotYoung and not self.plotPeak:
            self.gs = GridSpec(1, 2, width_ratios=ratios)

            self.cyclicStress(colorSeries, colorLinRange, plotPeak, plotFit)
            self.cyclicYoung(colorSeries)

        else:
            self.gs = GridSpec(1, 1)

            if self.plotStress and not self.plotYoung and not self.plotPeak:
                self.cyclicStress(colorSeries, colorLinRange, plotPeak, plotFit)

            if not self.plotStress and self.plotPeak and not self.plotYoung:
                self.cyclicPeak(colorSeries)

            if not self.plotStress and not self.plotPeak and self.plotYoung:
                self.cyclicYoung(colorSeries)

        self.fig.savefig(f'{self.filePath}.png', dpi=self.dpi)
        print(f'Chart saved at {self.fileFolder}')

        return self.fig

    def get_stressPeak(
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

    def cyclicStress(
            self,
            colorSeries, colorFit,  # Colors config
            plotPeak=bool, plotFit=bool  # Additional plots
    ):
        ax = self.fig.add_subplot(self.gs[:, 0])
        ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(1)

        ax.set_xlabel('Oscillation time (s)')
        ax.set_xticks([0, 0.5, 1, 1.5, 2])
        # ax.xaxis.set_minor_locator(MultipleLocator(0.25))
        ax.set_xlim([0, 2])

        ax.set_ylabel('Stress (kPa)')
        # ax.yaxis.set_major_locator(MultipleLocator(0.50))
        # ax.yaxis.set_minor_locator(MultipleLocator(0.25))
        ax.set_ylim([-0.25, self.stressData.max() + self.stressData.max() * 0.15])

        # Plot the linear range and its values
        ax.axvspan(self.i_linreg, self.f_linreg, alpha=0.07, color=colorFit)
        ax.text(self.i_linreg + 0.02, self.stressData.max() + self.stressData.max() * 0.1,
                f'Linear region', color=colorFit, weight='bold')
        ax.text(self.i_linreg - 0.165, -0.1, f'{self.i_linreg_pct:.1f}%', color=colorFit)
        ax.text(self.f_linreg + 0.020, -0.1, f'{self.f_linreg_pct:.1f}%', color=colorFit)

        # Plot the peak range
        x_peak, peak_mean, peak_std, peak_val = self.get_stressPeak(self.peakSize)
        if plotPeak:
            ax.axvspan(x_peak[-1][0], x_peak[-1][-1], alpha=0.2, color='mediumpurple')
            ax.text(x_peak[-1][-1] + 0.03, self.stressData.max() + self.stressData.max() * 0.1,
                    f'Peak region', color='mediumpurple', weight='bold')

        for i in np.arange(0, self.nCycles, 1):

            # Plot experimental data
            ax.scatter(
                np.append(self.timeData[i, :self.timeData.shape[1] // 2],
                          self.timeData[i, self.timeData.shape[1] // 2:]), self.stressData[i, :],
                label=f'#{i + 1} period', color=colorSeries, edgecolors='none', s=30, alpha=(0.8 - 0.25 * i))

            # If plot_peak is True, plot peak data
            if plotPeak:
                ax.scatter(
                    x_peak[i, :], peak_val[i, :],
                    color=colorSeries, edgecolors='indigo', linewidths=0.75, s=30, alpha=0.70)

            # If plot_fit is True, plot fitted curve
            self.slope_val, self.slope_std, popt = self.fitLinear(self.slope_val, self.slope_std, i)
            if plotFit:
                ax.plot(
                    self.timeData[i, self.i_index + 1:self.f_index + 1],
                    linear_reg(self.timeData[i, self.i_index + 1:self.f_index + 1], popt[0], popt[1]),
                    color=colorFit, alpha=(0.75 - 0.12 * i), lw=1.3)
                ax.scatter(
                    [self.timeData[i, self.i_index + 1], self.timeData[i, self.f_index + 1]],
                    [self.stressData[i, self.i_index + 1], self.stressData[i, self.f_index + 1]],
                    color=colorSeries, edgecolors=colorFit, linewidths=0.75, s=30, alpha=1)

        ax.legend(frameon=False)

    def cyclicPeak(
            self,
            color_series
    ):
        # if self.stress:
        #     ax = self.fig.add_subplot(self.gs[0, 1])
        # elif not self.stress:
        #     ax = self.fig.add_subplot(self.gs[0])

        if self.plotStress and not self.plotYoung:
            ax = self.fig.add_subplot(self.gs[0, 1])

        elif not self.plotStress and not self.plotYoung:
            ax = self.fig.add_subplot(self.gs[0])

        elif not self.plotStress and self.plotYoung:
            ax = self.fig.add_subplot(self.gs[1])

        else:
            ax = self.fig.add_subplot(self.gs[0, 1])

        x_peak, peak_mean, peak_std, peak_val = self.get_stressPeak(self.peakSize)

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

    def cyclicYoung(
            self,
            color_series
    ):
        if self.plotStress and not self.plotPeak:
            ax = self.fig.add_subplot(self.gs[0, 1])

        elif not self.plotStress and not self.plotPeak:
            ax = self.fig.add_subplot(self.gs[0])

        elif not self.plotStress and self.plotPeak:
            ax = self.fig.add_subplot(self.gs[1])

        else:
            ax = self.fig.add_subplot(self.gs[1, 1])

        period_array = np.arange(1, self.nCycles + 1, 1)

        for i in period_array - 1:
            self.slope_val, self.slope_std, popt = self.fitLinear(self.slope_val, self.slope_std, i)

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

    def fitLinear(
            self,
            slope,
            stdev,
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

        slope = np.append(slope, optimal[0])
        stdev = np.append(stdev, error[0])

        return slope, stdev, optimal

    def fitSinusoid(
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


class Sweep:
    def __init__(
            self,
            data_path,
            figure_size=(22, 15),
            dpi=300
    ):
        self.data_path = data_path
        self.fileTitle = os.path.basename(self.data_path[0]).split("/")[-1].split(".")[0]
        self.fileFolder = os.path.dirname(self.data_path[0])
        self.filePath = self.fileFolder + '\\' + self.fileTitle
        self.figure_size = figure_size
        self.dpi = dpi

        self.fig = plt.figure(self.fileTitle, figsize=(self.figure_size[0] * cm, self.figure_size[1] * cm))
        self.gs = GridSpec(1, 1)
        self.fig.subplots_adjust(hspace=0)

        # Collecting the data
        self.data, self.timeTotal, self.timeElement, self.strainStress, self.compViscosity, self.temperature, self.storageModulus, self.storageModulusErr, self.lossModulus, self.lossModulusErr, self.shearStress, self.frequency, self.angVeloc = None, None, None, None, None, None, None, None, None, None, None, None, None

    def plotStress(
            self,
            colorStorage='dodgerblue', colorLoss='hotpink'
    ):
        (self.shearStress, self.storageModulus, self.storageModulusErr,
         self.lossModulus, self.lossModulusErr) = self.getData('Shear Stress')
        ax = self.configPlot('Shear Stress')

        ax.errorbar(
            self.shearStress, self.storageModulus, yerr=self.storageModulusErr,
            label="G '",
            c=colorStorage, fmt='o', ms=6, alpha=0.9,
            ecolor=colorStorage, capthick=1, capsize=3, elinewidth=1)

        ax.errorbar(
            self.shearStress, self.lossModulus, yerr=self.lossModulusErr,
            label='G "',
            c='w', fmt='o', ms=6, alpha=0.9, mec=colorLoss,
            ecolor=colorLoss, capthick=1, capsize=3, elinewidth=1)

        ax.legend(ncol=1, frameon=False)
        self.fig.tight_layout()  # otherwise the right y-label is slightly clipped
        self.fig.savefig(f'{self.filePath}.png', dpi=self.dpi)
        print(f'Chart saved at {self.fileFolder}')

    def plotOscilatory(
            self,
            colorStorage='dodgerblue', colorLoss='hotpink'
    ):
        (self.angVeloc, self.storageModulus, self.storageModulusErr,
         self.lossModulus, self.lossModulusErr) = self.getData('Freq')
        ax = self.configPlot('Freq')

        ax.errorbar(
            self.angVeloc, self.storageModulus, yerr=self.storageModulusErr,
            label="G '",
            c=colorStorage, fmt='o', ms=6, alpha=0.9,
            ecolor=colorStorage, capthick=1, capsize=3, elinewidth=1)

        ax.errorbar(
            self.angVeloc, self.lossModulus, yerr=self.lossModulusErr,
            label='G "',
            c='w', fmt='o', ms=6, alpha=0.9, mec=colorLoss,
            ecolor=colorLoss, capthick=1, capsize=3, elinewidth=1)

        ax.legend(ncol=2, frameon=False)
        self.fig.tight_layout()  # otherwise the right y-label is slightly clipped
        self.fig.savefig(f'{self.filePath}.png', dpi=self.dpi)
        print(f'Chart saved at {self.fileFolder}')

    def plotRecSide(
            self,
            colorStorage='dodgerblue', colorLoss='hotpink'
    ):
        self.angVeloc, self.storageModulus, self.storageModulusErr, gIbef, self.lossModulus, self.lossModulusErr, gIIbef, storageModulus_aft, storageModulusErr_aft, gIaft, lossModulus_aft, lossModulusErr_aft, gIIaft = self.getDataRec()
        ax1, ax2 = self.configPlot('Recovery Side Freq')

        # TODO: deixar a visualização dos dados considerados cte mais bonita e copiar para o plotRecOverlap
        # Before breakage
        ax1.text(
            np.median(self.angVeloc), int(round(np.max(self.storageModulus) * 2, -4)),
            f'Before breakage', color='crimson', bbox=dict(facecolor='w', alpha=1, edgecolor='w', pad=0),
            horizontalalignment='center', verticalalignment='center')

        ax1.errorbar(
            self.angVeloc, self.storageModulus, yerr=self.storageModulusErr,
            label="G '",
            c=colorStorage, fmt='o', ms=6, alpha=0.9,
            ecolor=colorStorage, capthick=1, capsize=3, elinewidth=1)
        ax1.errorbar(
            self.angVeloc[:gIbef[1]], self.storageModulus[:gIbef[1]], yerr=self.storageModulusErr[:gIbef[1]],
            c=colorStorage, fmt='o', ms=6, alpha=0.9, mec='salmon',
            ecolor=colorStorage, capthick=1, capsize=3, elinewidth=1)

        ax1.errorbar(
            self.angVeloc, self.lossModulus, yerr=self.lossModulusErr,
            label='G "',
            c='w', fmt='o', ms=6, alpha=0.9, mec=colorLoss,
            ecolor=colorLoss, capthick=1, capsize=3, elinewidth=1)
        ax1.errorbar(
            self.angVeloc[:gIIbef[1]], self.lossModulus[:gIIbef[1]], yerr=self.lossModulusErr[:gIIbef[1]],
            c='w', fmt='o', ms=6, alpha=0.9, mec='salmon',
            ecolor=colorLoss, capthick=1, capsize=3, elinewidth=1)

        # After breakage
        ax2.text(
            np.median(self.angVeloc), int(round(np.max(self.storageModulus) * 2, -4)),
            f'After breakage', color='crimson', bbox=dict(facecolor='w', alpha=1, edgecolor='w', pad=0),
            horizontalalignment='center', verticalalignment='center')

        ax2.errorbar(
            self.angVeloc, storageModulus_aft, yerr=storageModulusErr_aft,
            label=f"G ' (RM: {100 * gIaft[0] / gIbef[0]:.1f}%)",
            c=colorStorage, fmt='o', ms=6, alpha=0.9,
            ecolor=colorStorage, capthick=1, capsize=3, elinewidth=1)
        ax2.errorbar(
            self.angVeloc[:gIaft[1]], storageModulus_aft[:gIaft[1]], yerr=storageModulusErr_aft[:gIaft[1]],
            c=colorStorage, fmt='o', ms=6, alpha=0.9, mec='salmon',
            ecolor=colorStorage, capthick=1, capsize=3, elinewidth=1)

        ax2.errorbar(
            self.angVeloc, lossModulus_aft, yerr=lossModulusErr_aft,
            label=f'G " (RM: {100 * gIIaft[0] / gIIbef[0]:.1f}%)',
            c='w', fmt='o', ms=6, alpha=0.9, mec=colorLoss,
            ecolor=colorLoss, capthick=1, capsize=3, elinewidth=1)
        ax2.errorbar(
            self.angVeloc[:gIIaft[1]], lossModulus_aft[:gIIaft[1]], yerr=lossModulusErr_aft[:gIIaft[1]],
            c='w', fmt='o', ms=6, alpha=0.9, mec='salmon',
            ecolor=colorLoss, capthick=1, capsize=3, elinewidth=1)

        # ax2.annotate('', xy=(self.angVeloc[gIIaft[1]], lossModulus_aft[gIIaft[1]]), xycoords='data',
        #              xytext=(self.angVeloc[0], lossModulus_aft[0]), textcoords='data',
        #              arrowprops=dict(arrowstyle='<->',
        #                              connectionstyle='bar',
        #                              ec='salmon',
        #                              shrinkA=3, shrinkB=3), zorder=1)

        # print(f'ANTES:\n{lossModulus_aft}\n'
        #       f'CTEs:\n{getCteRange(lossModulus_aft)}\n')

        ax2.legend(loc=3, ncol=1, frameon=False)
        self.fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.subplots_adjust(wspace=0, bottom=0.1)
        self.fig.savefig(f'{self.filePath}.png', dpi=self.dpi)
        print(f'Chart saved at {self.fileFolder}')

    def plotRecOverlap(
            self,
            colorStorage='dodgerblue', colorLoss='hotpink'
    ):
        self.angVeloc, self.storageModulus, self.storageModulusErr, gIbef, self.lossModulus, self.lossModulusErr, gIIbef, storageModulus_aft, storageModulusErr_aft, gIaft, lossModulus_aft, lossModulusErr_aft, gIIaft = self.getDataRec()
        ax = self.configPlot('Recovery Freq')

        # Before breakage
        ax.errorbar(0, 0, 0, alpha=0, label=f'Before breakage')
        ax.errorbar(
            self.angVeloc, self.storageModulus, yerr=self.storageModulusErr,
            label="G '",
            c=colorStorage, fmt='o', ms=6, alpha=0.9,
            ecolor=colorStorage, capthick=1, capsize=3, elinewidth=1)
        ax.errorbar(
            self.angVeloc, self.lossModulus, yerr=self.lossModulusErr,
            label='G "',
            c='w', fmt='o', ms=6, alpha=0.9, mec=colorStorage,
            ecolor=colorStorage, capthick=1, capsize=3, elinewidth=1)

        # After breakage
        ax.errorbar(0, 0, 0, alpha=0, label=f'After breakage')
        ax.errorbar(
            self.angVeloc, storageModulus_aft, yerr=storageModulusErr_aft,
            label=f"G ' (RM: {100 * gIaft[0] / gIbef[0]:.1f}%)",
            c=colorLoss, fmt='o', ms=6, alpha=0.9,
            ecolor=colorLoss, capthick=1, capsize=3, elinewidth=1)
        ax.errorbar(
            self.angVeloc, lossModulus_aft, yerr=lossModulusErr_aft,
            label=f'G " (RM: {100 * gIIaft[0] / gIIbef[0]:.1f}%)',
            c='w', fmt='o', ms=6, alpha=0.9, mec=colorLoss,
            ecolor=colorLoss, capthick=1, capsize=3, elinewidth=1)

        ax.legend(loc=3, ncol=2, frameon=False)
        self.fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.subplots_adjust(wspace=0, bottom=0.1)
        self.fig.savefig(f'{self.filePath}.png', dpi=self.dpi)
        print(f'Chart saved at {self.fileFolder}')

    def getData(
            self, mode
    ):
        nFiles = len(self.data_path)
        xData, gPrime, gDouble = np.array([]), np.array([]), np.array([])

        for file in range(nFiles):
            self.data = pd.read_excel(self.data_path[file])

            self.timeTotal = self.data['t in s'].to_numpy()
            self.timeElement = self.data['t_seg in s'].to_numpy()
            self.strainStress = self.data['ɣ in %'].to_numpy()
            self.compViscosity = self.data['|η*| in mPas'].to_numpy()
            self.temperature = self.data['T in °C'].to_numpy()

            if mode == 'Shear Stress':
                xData = self.data['τ in Pa'].to_numpy()
            if mode == 'Freq':
                freq = self.data['f in Hz'].to_numpy()
                xData = 2 * np.pi * freq

            gPrime = np.append(gPrime, self.data["G' in Pa"].to_numpy())
            gDouble = np.append(gDouble, self.data['G" in Pa'].to_numpy())

        gPrime = gPrime.reshape(
            nFiles, len(gPrime) // nFiles)
        gDouble = gDouble.reshape(
            nFiles, len(gDouble) // nFiles)

        return xData, gPrime.mean(axis=0), gPrime.std(axis=0), gDouble.mean(axis=0), gDouble.std(axis=0)

    def getDataRec(
            self, nPoints=31
    ):
        nFiles = len(self.data_path)

        x, gPrime_bef, gDouble_bef = np.array([]), np.array([]), np.array([])
        gPrime_aft, gDouble_aft = np.array([]), np.array([])

        for file in range(nFiles):
            self.data = pd.read_excel(self.data_path[file])

            self.timeTotal = self.data['t in s'].to_numpy()
            self.timeElement = self.data['t_seg in s'].to_numpy()
            self.strainStress = self.data['ɣ in %'].to_numpy()
            self.compViscosity = self.data['|η*| in mPas'].to_numpy()
            self.temperature = self.data['T in °C'].to_numpy()

            freq = self.data['f in Hz'].to_numpy()[:nPoints]
            x = 2 * np.pi * freq

            gPrime_bef = np.append(gPrime_bef, self.data["G' in Pa"].to_numpy()[:nPoints])
            gDouble_bef = np.append(gDouble_bef, self.data['G" in Pa'].to_numpy()[:nPoints])

            gPrime_aft = np.append(gPrime_aft, self.data["G' in Pa"].to_numpy()[-nPoints:])
            gDouble_aft = np.append(gDouble_aft, self.data['G" in Pa'].to_numpy()[-nPoints:])

        gPrime_bef = gPrime_bef.reshape(
            nFiles, len(gPrime_bef) // nFiles)
        gDouble_bef = gDouble_bef.reshape(
            nFiles, len(gDouble_bef) // nFiles)

        gPrime_aft = gPrime_aft.reshape(
            nFiles, len(gPrime_aft) // nFiles)
        gDouble_aft = gDouble_aft.reshape(
            nFiles, len(gDouble_aft) // nFiles)

        gIbef = gPrime_bef.mean(axis=0)
        gIbef_err = gPrime_bef.std(axis=0)
        gIbef_mean = meanCteRange(gIbef)

        gIIbef = gDouble_bef.mean(axis=0)
        gIIbef_err = gDouble_bef.std(axis=0)
        gIIbef_mean = meanCteRange(gIIbef)

        gIaft = gPrime_aft.mean(axis=0)
        gIaft_err = gPrime_aft.std(axis=0)
        gIaft_mean = meanCteRange(gIaft)

        gIIaft = gDouble_aft.mean(axis=0)
        gIIaft_err = gDouble_aft.std(axis=0)
        gIIaft_mean = meanCteRange(gIIaft)

        return x, gIbef, gIbef_err, gIbef_mean, gIIbef, gIIbef_err, gIIbef_mean, gIaft, gIaft_err, gIaft_mean, gIIaft, gIIaft_err, gIIaft_mean

    def configPlot(
            self,
            mode,
    ):
        plt.style.use('seaborn-v0_8-ticks')

        if 'Side' in mode:
            self.gs = GridSpec(1, 2)
        ax = self.fig.add_subplot(self.gs[:, 0])

        ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(1)
        ax.spines[['top', 'bottom', 'left', 'right']].set_color('dimgray')
        ax.set_yscale('log')
        ax.set_ylabel("Storage and Loss moduli (Pa)")
        ax.set_ylim([
            int(round(np.min(self.lossModulus) * 0.3, -2)),
            int(round(np.max(self.storageModulus) * 3, -4))])
        # ax.tick_params(axis='y', which='minor', labelsize=8)

        ax.set_xscale('log')
        if 'Freq' in mode:
            self.fig.suptitle(f'Frequency sweeps '
                              f'({self.fileTitle})', alpha=0.9)
            ax.set_xlabel('Angular velocity (rad/s)')
            # ax.set_xlim([self.angVeloc[0], self.angVeloc[-1] + 10])

        if mode == 'Shear Stress':
            self.fig.suptitle(f'Stress sweeps '
                              f'({self.fileTitle})', alpha=0.9)
            ax.set_xlabel('Shear stress (Pa)')
            ax.set_xlim([self.shearStress[0], self.shearStress[-1] + 10])

        if 'Recovery' in mode:
            self.fig.suptitle(f'Frequency sweeps | Modulus recovery '
                              f'({self.fileTitle})', alpha=0.9)

        if 'Side' in mode:
            self.fig.suptitle(f'Frequency sweeps | Modulus recovery '
                              f'({self.fileTitle})', alpha=0.9)
            ax2 = self.fig.add_subplot(self.gs[:, 1])
            ax.spines['right'].set_color('crimson')
            ax.spines['right'].set_linewidth(1.25)
            ax2.spines['left'].set_visible(False)
            ax2.spines[['top', 'bottom', 'right']].set_linewidth(1)
            ax2.spines[['top', 'bottom', 'right']].set_color('dimgray')

            ax2.tick_params(axis='y', which='both', left=False, labelleft=False, right=True, direction='in')

            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.set_ylim([
                int(round(np.min(self.lossModulus) * 0.3, -2)),
                int(round(np.max(self.storageModulus) * 3, -4))])

            return ax, ax2

        else:
            return ax


class General:
    def __init__(
            self,
            data_path,
            colorSeries,
            colorLinRange,
            figure_size=(22, 15),
            dpi=300
    ):
        self.data_path = data_path
        self.fileTitle = os.path.basename(self.data_path[0]).split("/")[-1].split(".")[0]
        self.fileFolder = os.path.dirname(self.data_path[0])
        self.filePath = self.fileFolder + '\\' + self.fileTitle
        self.figure_size = figure_size
        self.dpi = dpi

        self.fig = plt.figure(self.fileTitle, figsize=(self.figure_size[0] * cm, self.figure_size[1] * cm))
        self.gs = GridSpec(1, 1)
        self.fig.subplots_adjust(hspace=0)

        # TODO: a partir daqui rever
        # Collecting the data
        self.nFiles = len(self.data_path)
        self.data = pd.read_excel(self.data_path[0])

        self.head = list(self.data.columns)
        self.arrays = np.array([])

        ax = self.configPlot('Freq')

        gPrime = gPrime.reshape(
            nFiles, len(gPrime) // nFiles)

        ax.errorbar(
            self.angVeloc, self.storageModulus, yerr=self.storageModulusErr,
            label="G '",
            c=colorStorage, fmt='o', ms=6, alpha=0.9,
            ecolor=colorStorage, capthick=1, capsize=3, elinewidth=1)
        ax.legend(ncol=2, frameon=False)
        self.fig.tight_layout()  # otherwise the right y-label is slightly clipped

    def getCol(
            self,
            columnName=str
    ):
        return self.data[columnName].to_numpy()

    def plot(
            self
    ):
        pass


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

    Sweep.plotOscilatory(data)

    plt.show()
