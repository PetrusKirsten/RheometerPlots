import string
import numpy as np
import pandas as pd
from math import ceil

import matplotlib.colors as mcolors
from matplotlib import pyplot as plt, cm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from matplotlib.font_manager import FontProperties

from scipy.stats import f_oneway
from scipy.optimize import curve_fit
from statsmodels.stats.multicomp import pairwise_tukeyhsd

qntsMisc = ['SegIndex', 't in s', 'h in mm', 'T in °C', 't_seg in s']


def fonts(folder_path, s=13, m=15):
    font_path = folder_path + 'HelveticaNeueThin.otf'
    helvetica_thin = FontProperties(fname=font_path)

    font_path = folder_path + 'HelveticaNeueLight.otf'
    helvetica_light = FontProperties(fname=font_path)

    font_path = folder_path + 'HelveticaNeueMedium.otf'
    helvetica_medium = FontProperties(fname=font_path)

    font_path = folder_path + 'HelveticaNeueBold.otf'
    helvetica_bold = FontProperties(fname=font_path)

    plt.rc('font', size=s)  # controls default text sizes
    plt.rc('axes', titlesize=s)  # fontsize of the axes title
    plt.rc('axes', labelsize=s)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=s)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=s)  # fontsize of the tick labels
    plt.rc('legend', fontsize=s)  # legend fontsize
    plt.rc('figure', titlesize=m)  # fontsize of the figure title


def roundUp(value, mult=100.):
    mult = value
    result = ceil(value * 1/3 / mult) * mult
    return round(result, ndigits=4)


def statisticalAnalysis(samples, which='pre', modulus="G'", param='k'):
    """
    Realiza o ANOVA sobre o parâmetro especificado ('k' ou 'n') da power law,
    e realiza o teste de Tukey para comparar as formulações.

    :param samples: lista de objetos com .formulation e .powerLaw()
    :param which: 'pre' ou 'post'
    :param modulus: "G'" ou 'G"'
    :param param: 'k' ou 'n' — parâmetro a ser extraído
    :return: DataFrame dos parâmetros, resultado do ANOVA e resultados do Tukey
    """

    index_map = {'k': 0, 'n': 1}
    param_index = index_map[param]

    registros = []

    for sample in samples:
        formulation = sample.label
        fitResults = sample.powerLaw(which, modulus, raw_data=True)

        for rep, (params, errors) in fitResults.items():
            if params is not None:
                valor = params[param_index]
                registros.append({'formulation': formulation, 'value': valor, 'replicate': rep})

    df_params = pd.DataFrame(registros)

    # ANOVA
    groups = df_params.groupby('formulation')['value'].apply(list)
    anova = f_oneway(*groups)

    print(f"\n⚖️ ANOVA for '{param}':")
    print(f"F = {anova.statistic:.3f}, p = {anova.pvalue:.5f}")

    # Teste de Tukey
    tukey = pairwise_tukeyhsd(df_params['value'], df_params['formulation'], alpha=0.05)

    print("\n🔮 Tukey test results:")
    print(tukey.summary())

    return df_params, anova, tukey


def lettersTukey(tukey_result):

    def sortByNumber(group_dict):

        def extractNumber(key):
            try:
                return int(key.split('CL')[-1].strip())
            except (IndexError, ValueError):
                return float('inf')  # Coloca no final se não seguir o padrão

        return dict(sorted(group_dict.items(), key=lambda item: extractNumber(item[0])))

    results = pd.DataFrame(tukey_result._results_table.data[1:], columns=tukey_result._results_table.data[0])

    # Make dict to check if two groups are significantly different
    reject_dict = {
        frozenset([row['group1'], row['group2']]): row['reject']
        for _, row in results.iterrows()
    }

    # Get all unique group names
    groups = sorted(set(results['group1']) | set(results['group2']))

    # Initialize: letter_groups will hold sets of group names per letter
    letter_groups = []

    for group in groups:
        for letter_group in letter_groups:
            # If group is NOT significantly different from all members of this letter_group
            if all(not reject_dict.get(frozenset([group, other]), False) for other in letter_group):
                letter_group.add(group)
        # Start a new letter group just for this group (always added)
        letter_groups.append({group})

    # Remove duplicates: consolidate identical sets
    unique_groups = []
    for g in letter_groups:
        if not any(g <= existing for existing in unique_groups):
            unique_groups.append(g)

    # Assign letters (possibly multiple per group)
    group_to_letters = {group: '' for group in groups}
    for i, group_set in enumerate(unique_groups):
        letter = string.ascii_uppercase[i]
        for group in group_set:
            group_to_letters[group] += letter

    return sortByNumber(group_to_letters)


class OoRecovery:
    qntsOFS = ['f in Hz', "G' in Pa", 'G" in Pa', 'tan(δ) in -']  # '|G*| in Pa', 'tan(δ) in -', '|η*| in mPas']

    def __init__(self, paths, label, color):
        self.paths, self.label, self.color = paths, label, color

        self.rawPre, self.rawPost = self._get_data()
        # self.nReplicates = self.dataRawPre['Sample'].nunique()

        self.meanPre = self.rawPre.groupby('f in Hz')[self.qntsOFS].agg(['mean', 'std']).reset_index()
        self.meanPost = self.rawPost.groupby('f in Hz')[self.qntsOFS].agg(['mean', 'std']).reset_index()

    def _get_data(self):
        dfs_pre, dfs_post = [], []

        for i, file in enumerate(self.paths, start=1):
            df = pd.read_excel(file)[self.qntsOFS].dropna().reset_index(drop=True)

            breakIndex = len(df) // 2
            df_1, df_2 = df.iloc[:breakIndex].reset_index(drop=True), df.iloc[breakIndex:].reset_index(drop=True)

            df_1['Sample'], df_2['Sample'] = int(i), int(i)
            dfs_pre.append(df_1), dfs_post.append(df_2)

        return pd.concat(dfs_pre, ignore_index=True), pd.concat(dfs_post, ignore_index=True)

    def powerLaw(self, which, modulus, raw_data=False, n_points=16):
        """
        :param which: 'pre' ou 'post'
        :param modulus: "G'" ou 'G"'
        :param raw_data: Se True, aplica o ajuste para cada replicata dos dados brutos
        :param n_points: Quantos pontos usar para o ajuste (tipicamente os de baixa frequência)
        :return:
            - Se use_raw=False: (params, errors) tuple para dados médios
            - Se use_raw=True: dict {replicate: (params, errors)}
        """

        def function_powerLaw(frequency, k, n):
            return k * frequency ** n

        data = self.rawPre if (which == 'pre' and raw_data) else \
            self.rawPost if (which == 'post' and raw_data) else \
                self.meanPre if which == 'pre' else self.meanPost

        results = {}

        if raw_data:
            replicates = data['Sample'].unique()
            for rep in replicates:
                df_rep = data[data['Sample'] == rep].iloc[:n_points]

                x = df_rep['f in Hz'].values
                y = df_rep[f"{modulus} in Pa"].values

                try:
                    params = curve_fit(function_powerLaw, x, y)
                    stdev = np.sqrt(np.diag(params[1]))
                    results[rep] = (params[0], stdev)

                except Exception as e:
                    print(f"[!] Ajuste falhou para replicata {rep}: {e}")
                    results[rep] = (None, None)

            return results

        else:
            x = data['f in Hz']['mean'][:n_points]
            y = data[f"{modulus} in Pa", 'mean'][:n_points]

            params = curve_fit(function_powerLaw, x, y)
            stdev = np.sqrt(np.diag(params[1]))

            return params[0], stdev


class OoThixo:
    qntsFlow = ['SegIndex', 't in s', 't_seg in s', 'τ in Pa', 'η in mPas']

    def __init__(self, paths, label, color):
        self.paths, self.label, self.color = paths, label, color

        self.dataRaw = self._get_data()
        self.nReplicates = self.dataRaw['Sample'].nunique()

        self.dataMean = self.dataRaw.groupby('SegIndex')[self.qntsFlow[1:]].agg(['mean', 'std']).reset_index()
        # self.dataMean = self.dataMean.sort_values(by=(['t in s', 'mean']), inplace=True)

    def _get_data(self):

        def trim(dataframes):
            """
            :type dataframes: list
            :rtype: list
            """
            min_size = min(dataframe.shape[0] for dataframe in dataframes)

            return [dataframe.iloc[:min_size].reset_index(drop=True) for dataframe in dataframes]

        dfs = []

        for i, file in enumerate(self.paths, start=1):
            df = (
                pd.read_excel(file)[self.qntsFlow]
                .dropna()
                .query("`τ in Pa` != 0")
                .pipe(lambda d: d[d['SegIndex'].str.startswith('3|')])
                # .drop(columns='SegIndex')
                .pipe(lambda d: d.assign(**{"t in s": d["t in s"] - d["t in s"].min()}))
                .reset_index(drop=True)
            )
            df['Sample'] = int(i)
            dfs.append(df)

        return pd.concat(trim(dfs), ignore_index=True)

    def _powerLaw(self, modulus):
        """
        :type modulus: str
        :rtype: object
        """

        x, y = self.dataMean['f in Hz'][:16], None

        if modulus == "G'":
            y = self.dataMean["G' in Pa", 'mean'][:16]

        elif modulus == 'G"':
            y = self.dataMean['G" in Pa', 'mean'][:16]

        self.fitting = curve_fit(function_powerLaw, x, y)

        return self.fitting[0], self.fitting[1], np.sqrt(np.diag(self.fitting[1]))


class OoFlow:
    qntsFlow = ['SegIndex', 'ɣ̇ in 1/s', 'τ in Pa', 'η in mPas']

    def __init__(self, paths, label, color):
        self.paths, self.label, self.color = paths, label, color

        self.dataRaw = self._get_data()
        self.nReplicates = self.dataRaw['Sample'].nunique()

        self.dataMean = self.dataRaw.groupby('SegIndex')[self.qntsFlow[1:]].agg(['mean', 'std']).reset_index()

    def _get_data(self):

        def trim(dataframes):
            """
            :type dataframes: list
            :rtype: list
            """
            min_size = min(dataframe.shape[0] for dataframe in dataframes)

            return [dataframe.iloc[:min_size].reset_index(drop=True) for dataframe in dataframes]

        dfs = []

        for i, file in enumerate(self.paths, start=1):
            df = (
                pd.read_excel(file)[self.qntsFlow]
                .dropna()
                .query("`τ in Pa` != 0")
                .pipe(lambda d: d[d['SegIndex'].str.startswith('4|')])
                .reset_index(drop=True)
            )
            df['Sample'] = str(i)
            dfs.append(df)

        return pd.concat(trim(dfs), ignore_index=True)

    def _powerLaw(self, modulus):
        """
        :type modulus: str
        :rtype: object
        """

        x, y = self.dataMean['f in Hz'][:16], None

        if modulus == "G'":
            y = self.dataMean["G' in Pa", 'mean'][:16]

        elif modulus == 'G"':
            y = self.dataMean['G" in Pa', 'mean'][:16]

        # self.fitting = curve_fit(function_powerLaw, x, y)

        return self.fitting[0], self.fitting[1], np.sqrt(np.diag(self.fitting[1]))


def plotOFS(samples, save=False):
    def configFigure(width=4000, heigth=4000, dpi=300):

        def configAxes(ax, xLabel, yLabel, xLim=(0.08, 140), yLim=(1, 10 ** 4), axisColor='#303030'):

            ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(.75)
            ax.spines[['top', 'bottom', 'left', 'right']].set_color(axisColor)
            ax.tick_params(axis='both', which='both', colors=axisColor)

            ax.grid(True, which='major', axis='y', linestyle='-', linewidth=.75, color='lightgray', alpha=.5)
            ax.grid(True, which='minor', axis='y', linestyle='-', linewidth=.5, color='lightgray', alpha=.5)

            ax.set_xlabel(f'{xLabel}', color=axisColor), ax.set_ylabel(f'{yLabel}', color=axisColor)
            ax.set_xscale('log'), ax.set_yscale('log')
            ax.set_xlim(xLim), ax.set_ylim(yLim)

            if xLabel == '':
                ax.set_xticklabels([])
                ax.tick_params(axis='x', which='both', direction='in')

            if yLabel == '':
                ax.set_yticklabels([])
                ax.tick_params(axis='y', which='both', direction='in')

        fonts('C:/Users/petrus.kirsten/AppData/Local/Microsoft/Windows/Fonts/')
        plt.style.use('seaborn-v0_8-ticks')

        figure = plt.figure(figsize=(width / dpi, heigth / dpi), facecolor='snow')
        figure.canvas.manager.set_window_title(windowTitle)

        gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
        axes = [figure.add_subplot(gs[line, column]) for line in range(2) for column in range(2)]

        configAxes(axes[0], '', 'Elastic modulus')
        configAxes(axes[1], '', '')
        configAxes(axes[2], 'Frequency (Hz)', 'Viscous modulus')
        configAxes(axes[3], 'Frequency (Hz)', '')

        return figure, axes, dpi

    def addLegend(ax):

        legend = ax.legend(
            loc='lower right',
            fancybox=False,
            frameon=True,
            framealpha=0.9,
            fontsize=11,
            markerscale=1.3)

        legend.get_frame().set_facecolor('w')
        legend.get_frame().set_edgecolor('whitesmoke')

    def addMarkers(ax, label, color, x, y, yerr):

        lowError, highError = np.abs(yerr / y) <= 1.5, np.abs(yerr / y) > 1.5
        transp = .5

        # Plot error lines
        # markers with relative error <= 50%
        ax.errorbar(
            x[lowError], y[lowError], yerr[lowError],
            label='', fmt='none', markersize=0,
            color=color, mfc=color,
            alpha=transp,
            mec='w', mew=.75,
            capsize=2.5, capthick=1, linestyle='', lw=1,
            zorder=2)
        # markers with relative error > 50%
        ax.errorbar(
            x[highError], y[highError], yerr[highError],
            label='', fmt='none', markersize=0,
            color=color, mfc=color,
            alpha=transp / 10,
            mec='w', mew=.75,
            capsize=2.5, capthick=1, linestyle='', lw=1,
            zorder=2)

        # Plot markers
        # markers with relative error <= 50%
        ax.errorbar(
            x[lowError], y[lowError], 0,
            fmt='o', markersize=6,
            label=label,
            color=color, mfc=color,
            alpha=transp,
            mec=color, mew=.75,
            capsize=0, capthick=0, linestyle='', lw=0,
            zorder=3)
        # markers with relative error > 50%
        ax.errorbar(
            x[highError], y[highError], 0,
            fmt='o', markersize=6,
            label='',
            color=color, mfc=color,
            alpha=transp / 10,
            mec=color, mew=.75,
            capsize=0, capthick=0, linestyle='', lw=0,
            zorder=3)

    windowTitle = 'Viscoelastic recovery'
    fig, [axTL, axTR, axBL, axBR], resolution = configFigure()

    for sample in samples:
        pre, post = sample.meanPre, sample.meanPost
        l, c = sample.label, sample.color

        addMarkers(
            axTL, l, c,
            pre[('f in Hz', 'mean')], pre[("G' in Pa", 'mean')], pre[("G' in Pa", 'std')])
        addMarkers(
            axTR, l, c,
            post[('f in Hz', 'mean')], post[("G' in Pa", 'mean')], post[("G' in Pa", 'std')])

        addMarkers(
            axBL, l, c,
            pre[('f in Hz', 'mean')], pre[('G" in Pa', 'mean')], pre[('G" in Pa', 'std')])
        addMarkers(
            axBR, l, c,
            post[('f in Hz', 'mean')], post[('G" in Pa', 'mean')], post[('G" in Pa', 'std')])

    addLegend(axTL)

    plt.subplots_adjust(
        wspace=0.010, hspace=0.015,
        top=0.970, bottom=0.070,
        left=0.060, right=0.985)

    # plt.show()

    if save:
        fig.savefig(
            f'{windowTitle} ' + f'{l.split('CL')[0].strip()}' + ' CL' + '.svg',
            facecolor='w', dpi=resolution)


def plotBars(samples, param, lim, signifPre, signifPost, save=False):
    """
    :param signifPre:
    :param signifPost:
    :param lim:
    :param samples:
    :type param: str
    :param save:
    """

    def readData():
        def extract_params(state):
            return zip(*[
                (res[0][0], res[1][0], res[0][1], res[1][1])
                for res in (sample.powerLaw(state, "G'") for sample in samples)
            ])

        cacl = [int(sample.label.split()[-1]) for sample in samples]

        k_pre, kErr_pre, n_pre, nErr_pre = extract_params('pre')
        k_post, kErr_post, n_post, nErr_post = extract_params('post')

        if 'K' in param:
            return cacl, k_pre, kErr_pre, k_post, kErr_post
        else:
            return cacl, n_pre, nErr_pre, n_post, nErr_post

    def configFigure(width=2600, heigth=2500, dpi=300):

        def configAxes(ax, xLabel, yLabel, axisColor='#303030'):

            ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(.75)
            ax.spines[['top', 'bottom', 'left', 'right']].set_color(axisColor)
            ax.tick_params(axis='both', which='both', colors=axisColor)

            # ax.grid(True, which='major', axis='y', linestyle='-', linewidth=.75, color='lightgray', alpha=.5)
            # ax.grid(True, which='minor', axis='y', linestyle='-', linewidth=.5, color='lightgray', alpha=.5)

            xLim, yLim = (min(xData) - 1, max(xData) + 6), lim

            ax.set_xlabel(f'{xLabel}', color=axisColor), ax.set_ylabel(f'{yLabel}', color=axisColor)
            ax.set_xscale('linear'), ax.set_yscale('linear')
            ax.set_xlim(xLim)
            ax.xaxis.set_major_locator(MultipleLocator(7))
            # ax.xaxis.set_minor_locator(MultipleLocator(10))

            ax.set_ylim((0, yLim))
            ax.yaxis.set_major_locator(MultipleLocator(yLim / 5))
            ax.yaxis.set_minor_locator(MultipleLocator(yLim / 25))

            if xLabel == '':
                ax.set_xticklabels([])
                ax.tick_params(axis='x', which='both', direction='in')
            if yLabel == '':
                ax.set_yticklabels([])
                ax.tick_params(axis='y', which='both', direction='in')

        fonts('C:/Users/petrus.kirsten/AppData/Local/Microsoft/Windows/Fonts/')
        plt.style.use('seaborn-v0_8-ticks')

        figure = plt.figure(figsize=(width / dpi, heigth / dpi), facecolor='snow')
        figure.canvas.manager.set_window_title(windowTitle)
        figure.suptitle(f'{samples[0].label.split('CL')[0].strip()}' + ' CL')

        rows, columns = 1, 1
        gs = GridSpec(rows, columns, width_ratios=[1], height_ratios=[1])
        axes = [figure.add_subplot(gs[line, column]) for line in range(rows) for column in range(columns)]

        configAxes(
            axes[0],
            'CaCl$_2$ concentration (mM)',
            "Scale factor for elastic modulus,   $G_0'$ (Pa)" if 'K' in param else "Power law exponent,   $n'$")

        return figure, axes, dpi

    def addLegend(ax):

        legend = ax.legend(
            loc='upper left',
            fancybox=False,
            frameon=False,
            framealpha=0.9,
            fontsize=14,
            markerscale=1)

        legend.get_frame().set_facecolor('w')
        legend.get_frame().set_edgecolor('w')

    def drawData(ax, x, y, yerr, color, marker, label, significance, cmap_values):

        def addLetters():
            for letter, xi, yi, yerri in zip(significance.values(), x, y, yerr):

                yerri = yerri*10 if yerri <= 1 else yerri
                yerri = yerri*5 if yerri <= 1 else yerri
                ax.errorbar(
                    xi, yi, yerri,
                    label='', fmt='none', markersize=0,
                    color=color, mfc=color,
                    alpha=transp,
                    mec='w', mew=.75,
                    capsize=2.5, capthick=1, linestyle='-', lw=1,
                    zorder=2)

                offset = yi*.1 if xi == 0 else 0
                offset = -2*offset if 'After' in label else offset
                offset = lim*.01
                pct = '' #if 'Before' in label else f'\n{pct:.0f}%'
                dec = 2

                ax.text(
                    (xi + max(x)*.03), yi + offset,
                    f'{letter}\n'
                    r"$\mathbf{" + f'{yi:.0f} ~ ± ~ {1 * yerri:.0f} ~ Pa' + r"}$"
                    f'{pct}',
                    ha='left', va='bottom',
                    fontsize=12, color='k', alpha=0.85,
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.15', alpha=0.8),
                    zorder=4
                )

        def addColorBar():
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # Necessário para o ScalarMappable funcionar
            cbar = plt.colorbar(sm, ax=ax, pad=.02)
            cbar.set_label('Viscoelastic recovery', rotation=270, labelpad=15)
            ticks = cbar.get_ticks()
            tick_labels = [f'{tick:.0f}%' for tick in ticks]
            cbar.set_ticklabels(tick_labels)
            cbar.outline.set_linewidth(1)

        transp = 1.

        # Plot line
        ax.plot(
            x, y,
            marker='', linestyle='-', linewidth=1,
            color=color, alpha=transp - .2,
            zorder=1)

        # Plot error lines


        # Plot markers with colormap
        if cmap_values is not None:
            norm = mcolors.Normalize(vmin=min(cmap_values), vmax=max(cmap_values))
            norm = mcolors.Normalize(vmin=0, vmax=100)
            cmap = cm.get_cmap('RdYlGn')

            for xi, yi, val in zip(x, y, cmap_values):
                cor = cmap(norm(val))
                ax.errorbar(
                    xi, yi, 0,
                    fmt=marker, markersize=11,
                    label=None,
                    color=cor,
                    mfc=cor if 'Before' in label else 'w',
                    alpha=transp,
                    mec='#383838' if 'Before' in label else cor,
                    mew=1.0 if 'Before' in label else 1.75,
                    capsize=0, capthick=0, linestyle='', lw=0,
                    zorder=3)

            if 'After' in label:
                addColorBar()

        else:
            # Se não tiver colormap, usa cor fixa
            ax.errorbar(
                x, y, 0,
                fmt=marker, markersize=10,
                label=None,
                color=color, mfc=color if 'Before' in label else 'w',
                alpha=transp,
                mec='#383838' if 'Before' in label else color, mew=1,
                capsize=0, capthick=0, linestyle='', lw=0,
                zorder=3)

        # Plot for legend
        ax.errorbar(
            -10, -10, 0,
            fmt=marker, markersize=10,
            label=label,
            color=color, mfc=color if 'Before' in label else 'w',
            alpha=transp,
            mec='#383838' if 'Before' in label else color, mew=1,
            capsize=0, capthick=0, linestyle='', lw=0,
            zorder=3)

        addLetters()

    xData, yDataPre, yErrDataPre, yDataPost, yErrDataPost = readData()

    windowTitle = f'Power law fitting - {param}'
    fig, [axs], resolution = configFigure()

    recovery = (np.array(yDataPost) / np.array(yDataPre))  * 100

    for legend, yData, yDataErr, letters, cmap_vals in [
        ('Before shear', yDataPre, yErrDataPre, signifPre, None)]:
        # ('After shear', yDataPost, yErrDataPost, signifPost, None)]:

        drawData(
            ax=axs, x=xData, y=yData, yerr=yDataErr,
            color='#57cc99', marker='o', label=legend,
            significance=letters, cmap_values=None)

    # addLegend(axs)

    plt.tight_layout()
    # plt.show()

    if save:
        fig.savefig(
            f'{windowTitle} - ' + f'{samples[0].label.split('CL')[0].strip()}' + ' CL' + '.svg',
            facecolor='w', dpi=resolution)


def main():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    folderPath = "D:/Documents/GitHub/Rheometer-Plotting/data/by sample"

    filePath = [
        # St
        folderPath + "/10St/10_0WSt-viscRec_1.xlsx",
        folderPath + "/10St/10_0WSt-viscRec_2.xlsx",

        # St/CL_7
        folderPath + "/10St_CL_7/10_0St_CL-recovery-1.xlsx",
        # folderPath + "/10St_CL_7/10_0St_CL-recovery-2.xlsx",
        folderPath + "/10St_CL_7/10_0St_CL-recovery-3.xlsx",

        # St/CL_14
        folderPath + "/10St_CL_14/St_CL_14-viscoelasticRecovery-1.xlsx",
        folderPath + "/10St_CL_14/St_CL_14-viscoelasticRecovery-2.xlsx",
        folderPath + "/10St_CL_14/St_CL_14-viscoelasticRecovery-3.xlsx",
        folderPath + "/10St_CL_14/St_CL_14-viscoelasticRecovery-4.xlsx",

        # St/CL_28
        folderPath + "/10St_CL_28/St_CL_28-viscoelasticRecovery-1.xlsx",
        folderPath + "/10St_CL_28/St_CL_28-viscoelasticRecovery-2.xlsx",
        folderPath + "/10St_CL_28/St_CL_28-viscoelasticRecovery-3.xlsx",
        folderPath + "/10St_CL_28/St_CL_28-viscoelasticRecovery-4.xlsx",
    ]

    stCL_recovery = [
        OoRecovery(filePath[:2], 'St CL 0', '#E1C96B'),
        OoRecovery(filePath[2:4], 'St CL 7', '#FFE138'),
        OoRecovery(filePath[4:8], 'St CL 14', '#F1A836'),
        OoRecovery(filePath[8:], 'St CL 21', '#E36E34'),
    ]

    # plotOFS(stCL_recovery)

    # stCL_thixo = [
    #     OoThixo(filePath[:2], 'St CL 0', '#E1C96B'),
    #     OoThixo(filePath[2:4], 'St CL 7', '#FFE138'),
    #     OoThixo(filePath[4:8], 'St CL 14', '#F1A836'),
    #     OoThixo(filePath[8:], 'St CL 21', '#E36E34'),
    # ]

    stCL_flow = [
        OoFlow(filePath[:2], 'St CL 0', '#E1C96B'),
        OoFlow(filePath[2:4], 'St CL 7', '#FFE138'),
        OoFlow(filePath[4:8], 'St CL 14', '#F1A836'),
        OoFlow(filePath[8:], 'St CL 21', '#E36E34'),
    ]


if __name__ == '__main__':
    main()
