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


def formsPLot(row, col, titles):
    fonts(folder_path='C:/Users/petrus.kirsten/AppData/Local/Microsoft/Windows/Fonts/')
    ax = axs[row, col]
    ax.set_proj_type('ortho')
    ax.view_init(elev=13, azim=20, roll=0)
    if r < 1:
        ax.set_title(f'{titles[c]}', fontsize=10)
    plt.tick_params(axis='x', labelsize=10)
    plt.tick_params(axis='y', labelsize=10)
    plt.tick_params(axis='z', labelsize=10)
    for axis in ax.xaxis, ax.yaxis, ax.zaxis:
        axis.set_label_position('lower')
        axis.set_ticks_position('lower')
    ax.set_xlim([-6, 20])
    ax.set_xticks([0, 14])
    ax.set_xticklabels(['0 mM Ca$^{2+}$', '14 mM Ca$^{2+}$'], rotation=0)
    ax.set_ylim([-2, 12])
    ax.set_yticks([0, 5.0, 10.0])
    ax.set_yticklabels(['0% WSt', '5% WSt', '10% WSt'], rotation=30)
    ax.set_zlim([0, 1.25])
    ax.set_zticks([0, 0.5, 1.0])
    ax.set_zticklabels(['0% kCar', '5.0% kCar', '1.0% kCar'], rotation=40)
    if r > 0:
        ax.set_zticklabels(['0% iCar', '5.0% iCar', '1.0% iCar'], rotation=40)
    ax.grid(False)
    edgeColor = 'k'
    ax.scatter(x[:2], y[:2], z[:2],
               c=z[:2], cmap='Blues', marker='o', alpha=1, s=120,
               edgecolors=edgeColor, linewidths=0.75,
               zorder=4)
    for i, j, k, h in zip(x[:2], y[:2], z[:2], z2[:2]):
        ax.plot([i, i], [j, j], [k, h],
                color='slategrey', alpha=0.9, lw=1.25,
                zorder=1)

    ax.scatter(x[2:-2], y[2:-2], z[2:-2],
               c=z[2:-2], cmap='Greens', marker='o', alpha=1, s=120,
               edgecolors=edgeColor, linewidths=0.75,
               zorder=4)
    for i, j, k, h in zip(x[2:-2], y[2:-2], z[2:-2], z2[2:-2]):
        ax.plot([i, i], [j, j], [k, h],
                color='seagreen', alpha=0.9, lw=1.25,
                zorder=1)

    ax.scatter(x[-2:], y[-2:], z[-2:],
               c=z[-2:], cmap='Wistia', marker='o', alpha=1, s=120,
               edgecolors=edgeColor, linewidths=0.75,
               zorder=4)
    for i, j, k, h in zip(x[-2:], y[-2:], z[-2:], z2[-2:]):
        ax.plot([i, i], [j, j], [k, h],
                color='chocolate', alpha=0.8, lw=1.25,
                zorder=1)

    ax.scatter(x14[:2], y14[:2], z14[:2],
               c=z14[:2], cmap='Blues', marker='s', alpha=1, s=120,
               edgecolors=edgeColor, linewidths=0.75,
               zorder=4)
    for i, j, k, h in zip(x14[:2], y14[:2], z14[:2], z214[:2]):
        ax.plot([i, i], [j, j], [k, h],
                color='slategrey', alpha=0.9, lw=1.25,
                zorder=1)

    ax.scatter(x14[2:-2], y14[2:-2], z14[2:-2],
               c=z14[2:-2], cmap='Greens', marker='s', alpha=1, s=120,
               edgecolors=edgeColor, linewidths=0.75,
               zorder=4)
    for i, j, k, h in zip(x14[2:-2], y14[2:-2], z14[2:-2], z214[2:-2]):
        ax.plot([i, i], [j, j], [k, h],
                color='seagreen', alpha=0.9, lw=1.25,
                zorder=1)

    ax.scatter(x14[-2:], y14[-2:], z14[-2:],
               c=z14[-2:], cmap='Wistia', marker='s', alpha=1, s=120,
               edgecolors=edgeColor, linewidths=0.75,
               zorder=4)
    for i, j, k, h in zip(x14[-2:], y14[-2:], z14[-2:], z214[-2:]):
        ax.plot([i, i], [j, j], [k, h],
                color='chocolate', alpha=0.8, lw=1.25,
                zorder=1)

    ax.invert_xaxis()
    # ax.invert_yaxis()
    # ax.invert_zaxis()
    paneColor = 'floralwhite'
    edgeColor = 'k'
    if r > 0:
        paneColor = 'mintcream'
    ax.xaxis.set_pane_color(paneColor)
    ax.yaxis.set_pane_color(paneColor)
    ax.zaxis.set_pane_color(paneColor)
    ax.xaxis.pane.set_edgecolor(edgeColor)
    ax.yaxis.pane.set_edgecolor(edgeColor)
    ax.zaxis.pane.set_edgecolor(edgeColor)


cm = 1 / 2.54  # centimeters in inches
plt.style.use('seaborn-v0_8-ticks')

# [starch, car, Ca2+]
formulas0mM = np.array([
    [10, 0, 0, 0],  # pst10_0dht
    [10, 1, 0, 0],  # pst10_1car_0dht

    [5, 0, 0, 0],  # pst5_0dht
    [5, 0.5, 0, 0],  # pst5_05car_0dht

    [0, 1.0, 0, 0],  # car1
    [0, 0.5, 0, 0],  # car05
])
x = formulas0mM[:, 3]
y = formulas0mM[:, 0]
z = formulas0mM[:, 1]
z2 = np.ones(shape=x.shape) * min(z)

formulas14mM = np.array([
    [10, 0, 0, 14],  # pst10_0dht
    [10, 1, 0, 14],  # pst10_1car_0dht

    [5, 0, 0, 14],  # pst5_0dht
    [5, 0.5, 0, 14],  # pst5_05car_0dht

    [0, 1.0, 0, 14],  # car1
    [0, 0.5, 0, 14]  # car05
])
x14 = formulas14mM[:, 3]
y14 = formulas14mM[:, 0]
z14 = formulas14mM[:, 1]
z214 = np.ones(shape=x.shape) * min(z)

rows, cols = 2, 3
fig, axs = plt.subplots(rows, cols, figsize=(35 * cm, 20 * cm),
                        subplot_kw={'projection': '3d'}, constrained_layout=False)

left = 0       # the left side of the subplots of the figure
right = 1      # the right side of the subplots of the figure
bottom = 0.02  # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.0   # the amount of width reserved for blank space between subplots
hspace = 0.0   # the amount of height reserved for white space between subplots

fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
t = ['Native wheat starch.', 'Modified wheat starch by 1 h DHT.', 'Modified wheat starch by 2 h DHT.']  # columns title
for r in range(0, rows):
    for c in range(0, cols):
        formsPLot(r, c, t)
        formsPLot(r, c, t)
fig.suptitle(f'Hydrogels formulations. '
             f'Total: {len(formulas0mM) * rows * cols + len(formulas14mM) * rows * cols}.', fontsize=13)
plt.show()
fig.savefig(f'HydrogelsFormulation.png', dpi=600)
