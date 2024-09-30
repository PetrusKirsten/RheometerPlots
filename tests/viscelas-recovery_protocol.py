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


# Plots configs
plt.style.use('seaborn-v0_8-ticks')
# self.gs = GridSpec(1, 1)
# ax1 = self.fig.add_subplot(self.gs[:, 0])
# self.fig.suptitle(f'Dynamic compression - Full oscillation '
#                   f'({self.fileTitle})', alpha=0.9)

fonts(folder_path='C:/Users/petrus.kirsten/AppData/Local/Microsoft/Windows/Fonts/')
fig, axes = plt.subplots(figsize=(12, 6), facecolor='w', ncols=2)
fig.tight_layout()

# Left axis configs
ax1 = axes[0]
ax1.set_xlabel('Time (s)')
# ax1.set_xlim([0, 2 * self.nCycles])
ax1.set_ylabel('Stress (kPa)')
# ax1.set_ylim([self.stressData.min() - self.stressData.min() * 0.1,
#               self.stressData.max() + self.stressData.max() * 0.1])
# ax1.yaxis.set_major_locator(MultipleLocator(0.50))
ax1.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)
# ax1.yaxis.set_minor_locator(MultipleLocator(0.25))

# Experimental data


# Right plot
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.spines['left'].set_color('r')
ax2.spines['right'].set_color('r')
# ax2.set_xlim([0, 2 * self.nCycles])

ax2.set_ylabel('Strain (%)', color='r')
ax2.tick_params(axis='y', labelcolor='r', colors='r')
ax2.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)
# ax2.set_ylim([self.heightData.min() - self.heightData.min() * 0.1,
#               self.heightData.max() + self.heightData.max() * 0.1])
# ax2.yaxis.set_major_locator(MultipleLocator(0.5))

ax1.set_ylabel('Stress (kPa)', color='b')
ax1.tick_params(axis='y', labelcolor='b', colors='b')

# Experimental data

axes[1].legend(loc=4, frameon=False)
# plt.subplots_adjust(wspace=0, top=0.85, bottom=0.1, left=0.18, right=0.95)
plt.subplots_adjust(wspace=0, bottom=0.1, left=0.2)

filename = 'plot_sfe'
plt.tight_layout()
plt.show()
# plt.savefig(filename + '.png', facecolor='w', dpi=300)
