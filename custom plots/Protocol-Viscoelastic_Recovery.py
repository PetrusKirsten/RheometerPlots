import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle


def fonts(folder_path, small=10, medium=12):
    """Configures font properties for plots."""
    plt.rc('font', size=small)
    plt.rc('axes', titlesize=medium)
    plt.rc('axes', labelsize=medium)
    plt.rc('xtick', labelsize=medium)
    plt.rc('ytick', labelsize=medium)
    plt.rc('legend', fontsize=small)
    plt.rc('figure', titlesize=medium)


def genThixo(n=20, b=2, p=2.5, d=0.5):
    """Generates a series of n points with thixotropic behavior and exponential decay."""
    initial = np.full(0, b)
    growth = np.linspace(b, p, 1)
    decay_length = n - len(initial) - len(growth)
    decay = p * np.exp(-d * np.arange(decay_length))
    final_decay = b + (decay - b)
    series = np.concatenate([growth, final_decay])
    return series


def genShearStress(shearRate, k=1, n=0.5):
    """Calculates shear stress for pseudoplastic material using power law: τ = k * g^n."""
    return k * (shearRate ** n)


def plotFreqSweep(
        ax, x, y, base, markerSize,
        title, textConfig, textLabel, textCoord, rectConfig,
        tickRight=False):
    """Plots the Oscillatory Frequency Sweep Assay."""
    ax.set_title(title, size=10)
    ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)
    ax.set_xticks([])
    ax.set_xlim([-3, 10.5])

    ax.set_yticks([1, 10])
    ax.tick_params(axis='y', colors='mediumaquamarine')
    ax.set_yticklabels(['$0.1\,Hz$', '$100\,Hz$'], size=9.5, color='mediumseagreen')
    ax.set_ylim([0, 10.5])
    if tickRight:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')

    ax.plot(x, y, color='mediumaquamarine', lw=1.25, alpha=0.85, label='ω', zorder=1)
    ax.scatter(*genStoMod(base, 10), lw=.5, c='dodgerblue', edgecolor='k', s=markerSize, marker='v', label="G'", zorder=2)
    ax.scatter(*genLosMod(base - 1, 10), lw=.5, c='lightskyblue', edgecolor='k', s=markerSize, marker='^', label='G"', zorder=2)

    ax.text(textCoord[0], textCoord[1], s=textLabel, **textConfig)
    rect = Rectangle(*rectConfig, linewidth=1, edgecolor='k', facecolor='whitesmoke', alpha=0.75, zorder=1)
    ax.add_patch(rect)

    ax.grid(ls='--', color='mediumaquamarine', alpha=0.5, zorder=0)
    legendLabel(ax)


def genStoMod(base, n=10):
    """Generates G' modulus data for the plots."""
    xMod = np.linspace(0, 10, n + 1)
    initialPoints = base + np.random.uniform(-0.2, 0.1, n - 4)
    yStoMod = np.concatenate([
        [base, base],  # Os primeiros 2 pontos paralelos ao eixo X
        initialPoints,  # Pontos ligeiramente variáveis
        [base - 0.5, base - 1, base - 2]  # Últimos pontos suavemente decrescendo
    ])
    return xMod, yStoMod


def genLosMod(base, n=10):
    """Generates G" modulus data for the plots."""
    xMod = np.linspace(0, 10, n + 1)
    base = base / 1.5
    initialPoints = base + np.random.uniform(-0.1, 0.1, n - 4)
    yLossMod = np.concatenate([
        [base, base],  # Os primeiros 2 pontos paralelos ao eixo X
        initialPoints,  # Pontos ligeiramente variáveis
        [base + 0.5, base + 1, base + 2]  # Últimos pontos suavemente crescendo
    ])
    return xMod, yLossMod


def plotShearAssay(ax, x, y, textSize):
    """Plots the Flow Shearing Assay."""
    ax.set_title('Flow shearing assay.', size=10)
    ax.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)
    ax.set_xticks([])
    ax.set_xlim([-10, 10])
    ax.set_xlabel('Time')
    ax.set_yticks([])
    ax.set_ylim([-0.15, 3.3])

    y_shear_scte = genThixo()
    x_shear_scte = np.linspace(0, 10, y_shear_scte.shape[0])
    x_shear_rcte = np.linspace(-10, 0, 20)
    y_shear_rcte = np.full(20, y[-1])
    x_shear_s = np.linspace(x[0], 10, 20)
    y_shear_s = genShearStress(x_shear_s, k=0.65, n=0.5)

    ax.text(x_shear_rcte[len(x_shear_rcte) // 2], 2.9, 'Constant strain rate\n$\dot{γ}=100 \,\,\, s^{-1}$', horizontalalignment='center', verticalalignment='bottom', color='k', size=textSize)
    ax.text(x[len(x) // 2], 2.9, 'Decreasing strain rate.\n$100 > \dot{γ} > 0.1 \,\,\, s^{-1}$', horizontalalignment='center', verticalalignment='bottom', color='k', size=textSize)

    ax.plot(x_shear_rcte, y_shear_rcte, color='orange', lw=1.25, alpha=0.75, zorder=2)
    ax.plot(-x + 10, y, color='orange', lw=1.25, alpha=0.75, label='$\dot{γ}$', zorder=2)
    ax.scatter(x_shear_scte - 10, y_shear_scte, lw=.5, c='hotpink', edgecolor='k', s=30, marker='o', label='σ', zorder=3)
    ax.scatter(x_shear_s, y_shear_s, lw=.5, c='hotpink', edgecolor='k', s=30, marker='o', zorder=3)
    legendLabel(ax)


def legendLabel(ax):
    """Applies consistent styling to legends in plots."""
    legend = ax.legend(loc=4, frameon=True, framealpha=1, fancybox=False, scatterpoints=3)
    legend.get_frame().set_facecolor('w')
    legend.get_frame().set_edgecolor('whitesmoke')


# Main Plotting Configuration
def mainPlot(filename):
    fonts(folder_path='C:/Users/petrus.kirsten/AppData/Local/Microsoft/Windows/Fonts/')
    plt.style.use('seaborn-v0_8-ticks')
    fig, axes = plt.subplots(figsize=(16, 16 // 3), facecolor='w', ncols=3)
    fig.suptitle('Rheometry assay protocol to evaluate viscoelastic recovery.\n\n\n')
    marker_size = 50

    # Common parameters
    xFreq = np.linspace(0, 10, 1000)
    yFreq = np.ceil(xFreq)
    text_coord = (-1.5, 10.1)
    text_label = 'Rest and\nSet $T=37\,^oC$\nfor $180\,\,s$'
    text_properties = {'horizontalalignment': 'center', 'verticalalignment': 'top', 'color': 'k', 'size': 9.2}
    rect_properties = [(-3, 0), 3, 10.5]

    # Plot 1: Oscillatory Frequency Sweep Assay
    plotFreqSweep(
        axes[0], xFreq, yFreq, 8, 50,
        'Oscillatory frequency sweep assay.',
        text_properties, text_label, text_coord, rect_properties)

    # Plot 2: Flow Shearing Assay
    xShearRate = np.linspace(0, 10, 1000)
    yShearRate = np.ceil(xShearRate) / 3.5
    plotShearAssay(
        axes[1], xShearRate, yShearRate,
        textSize=9.2)

    # Plot 3: Oscillatory Frequency Sweep Assay Again
    plotFreqSweep(
        axes[2], xFreq, yFreq, 6, 50,
        'Oscillatory frequency sweep assay again.',
        text_properties, text_label, text_coord, rect_properties,
        tickRight=True)

    plt.subplots_adjust(wspace=0.0, top=0.875, bottom=0.05, left=0.05, right=0.95)
    fig.savefig(f'{filename}.png', facecolor='w', dpi=600)
    plt.show()


# Run the main plotting function
mainPlot('viscoelastic-recovery_protocol')
