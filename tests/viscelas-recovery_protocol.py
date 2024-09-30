import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle


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


fonts(folder_path='C:/Users/petrus.kirsten/AppData/Local/Microsoft/Windows/Fonts/')
# Plots configs
plt.style.use('seaborn-v0_8-ticks')
fig, axes = plt.subplots(figsize=(12, 6), facecolor='w', ncols=3)

# 1st plot configs
ax1 = axes[0]
ax1.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)
ax1.set_xticks([])
ax1.set_yticks([])

xFreq = np.linspace(0, 10, 1000)
yFreq = np.ceil(xFreq)  # np.ceil creates a step effect by rounding up

# Generate frequency values (log scale for realistic behavior)
xMod = np.logspace(-1, 2, 6)  # Frequencies from 0.1 to 100 (rad/s)
# Simulate Storage Modulus (Elastic Modulus) as a function of frequency
yStoMod = 2 + 200 * np.log10(xMod)  # Increases logarithmically with frequency
# Simulate Loss Modulus as a function of frequency
yLosMod = 2 * xMod / (1 + xMod ** 2)  # Peaks at intermediate frequencies
# TODO: arrumar dados do módulo elástico
ax1.plot(xFreq, yFreq,
         color='g', lw=1.5,
         label='Frequency')

(ax1.scatter(xMod, yStoMod,
             color='r', lw=1.5,
             label='Storage and Loss moduli'))

ax1.legend(loc=4, frameon=False)

# 2nd plot configs
ax2 = axes[1]
ax2.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)
ax2.set_xticks([])
ax2.set_xlabel('Time (s)')
ax2.set_yticks([])

# 3rd plot configs
ax3 = axes[2]
ax3.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)
ax3.set_xticks([])
ax3.set_yticks([])

# ax2.legend(loc=4, frameon=False)
plt.subplots_adjust(wspace=0.0, top=0.95, bottom=0.1, left=0.05, right=0.95)
plt.show()
filename = 'plot_sfe'
# plt.savefig(filename + '.png', facecolor='w', dpi=300)
