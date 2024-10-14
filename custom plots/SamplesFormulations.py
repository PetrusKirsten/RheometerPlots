import numpy as np
import matplotlib.pyplot as plt
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


def plotFormulas(ax, formulas, title, typeCar=None, paneColor='whitesmoke', edgeColor='dimgrey'):
    ax.set_proj_type('ortho')
    ax.view_init(elev=29 if typeCar else 90, azim=134 if typeCar else 90, roll=0 if typeCar else 90)
    ax.grid(False)
    for axis in ax.xaxis, ax.yaxis, ax.zaxis:
        axis.set_label_position('lower' if typeCar else 'lower')
        axis.set_ticks_position('lower' if typeCar else 'lower')
        axis.set_pane_color(paneColor)
        axis.pane.set_edgecolor(edgeColor)

    ax.set_title(title)
    # x axis
    ax.set_xlabel('DHT time')
    ax.set_xlim([3, -1])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['0 h', '1 h', '2 h'])
    # ax.invert_xaxis()
    # y axis
    ax.set_ylabel('Starch')
    ax.set_ylim([12, -2] if typeCar else [2, 13])
    ax.set_yticks([0, 5.0, 10.0] if typeCar else [5.0, 10.0])
    ax.set_yticklabels(['0%', '5%', '10%'] if typeCar else ['5%', '10%'])
    # ax.invert_yaxis()
    # z axis
    ax.set_zlabel(f'Carrageenan' if typeCar else '')
    ax.set_zlim([0, 1.25])
    ax.set_zticks([0, 0.5, 1.0] if typeCar else [])
    ax.set_zticklabels(['0%', '0.5%', '1.0%'] if typeCar else [])

    for formulation in formulas:
        dht, starch, carrageenan, cacl2 = formulation
        color = 'hotpink' if cacl2 else 'whitesmoke'
        markerFill = 'right' if cacl2 else 'left'
        labelLegend = '$Ca^{2+}$' if cacl2 else 'No CL'
        ax.plot(dht, starch, carrageenan,
                c=color, alpha=(0.4 + starch * 0.06), fillstyle=markerFill, ms=12, marker='o', mew=0.5, mec='k',
                label=labelLegend, zorder=2)
        # Add drop lines to the x-y plane
        ax.plot([dht, dht], [starch, starch], [0, carrageenan - 0.06],
                color='#383838', linestyle='-', alpha=(0.4 + starch * 0.06), lw=0.6,
                zorder=1)


# Define formulations
wStOnly = [
    [0, 10, 0, False], [0, 10, 0, True],
    [1, 10, 0, False], [1, 10, 0, True],
    [2, 10, 0, False], [2, 10, 0, True],
    [0, 5, 0, False], [0, 5, 0, True],
    [1, 5, 0, False], [1, 5, 0, True],
    [2, 5, 0, False], [2, 5, 0, True]
]
iCar = [
    [0, 10, 1, False], [0, 10, 1, True],
    [1, 10, 1, False], [1, 10, 1, True],
    [2, 10, 1, False], [2, 10, 1, True],
    [0, 5, 0.5, False], [0, 5, 0.5, True],
    [1, 5, 0.5, False], [1, 5, 0.5, True],
    [2, 5, 0.5, False], [2, 5, 0.5, True],
    [0, 0, 1, False], [0, 0, 1, True]
]
kCar = [
    [0, 10, 1, False], [0, 10, 1, True],
    [1, 10, 1, False], [1, 10, 1, True],
    [2, 10, 1, False], [2, 10, 1, True],
    [0, 5, 0.5, False], [0, 5, 0.5, True],
    [1, 5, 0.5, False], [1, 5, 0.5, True],
    [2, 5, 0.5, False], [2, 5, 0.5, True],
    [0, 0, 1, False], [0, 0, 1, True]
]

# Create figure and subplots
plt.style.use('seaborn-v0_8-ticks')
fig = plt.figure(figsize=(15, 5), constrained_layout=False)
fig.suptitle('Hydrogels formulations. 40 in total')
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')
# Adjust subplot layout
fig.subplots_adjust(
    left=0.01,  # Left side of the subplots
    right=0.99,  # Right side of the subplots
    bottom=0.02,  # Bottom of the subplots
    top=0.87,  # Top of the subplots
    wspace=0.05,  # Width reserved for blank space between subplots
    hspace=0.0  # Height reserved for blank space between subplots
)
# Plot formulations
plotFormulas(
    ax1, wStOnly,
    'Starch-only | 12 formulations.',
    paneColor='whitesmoke', edgeColor='gainsboro')
plotFormulas(
    ax2, kCar,
    'Kappa Carrageenan | 14 formulations.', 'Kappa Carrageenan',
    paneColor='floralwhite', edgeColor='antiquewhite')
plotFormulas(
    ax3, iCar,
    'Iota Carrageenan | 14 formulations.', 'Iota Carrageenan',
    paneColor='azure', edgeColor='paleturquoise')

# Display then save
filename = 'Hydrogels-Formulations-Plot'
plt.show()
fig.savefig(f'{filename}.png', dpi=600, bbox_inches='tight')
