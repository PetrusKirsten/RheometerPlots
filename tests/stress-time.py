import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

np.set_printoptions(threshold=np.inf)  # to print the entire array
cm = 1 / 2.54  # centimeters in inches


# Fonts in chart config
def fonts(small=8, medium=12, big=14, folder_path=str):
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


# Data config
df = pd.read_csv('../data/haribo-9v2.csv')
# print(df.to_string())

max_id = df['Fn in N'].idxmax()  # index of max force value
max_h = df['h in mm'].iloc[max_id]  # find the height where the force is max

array_df = df.drop(columns=['SegIndex']).to_numpy()
# print(array_df[:, 3])  # select values from 3rd column

n_points = 98
n_cycles = int(array_df[:, 3].shape[0] / n_points)

fn_seq = array_df[:, 0].reshape(n_cycles, n_points)
h_seq = array_df[:, 1].reshape(n_cycles, n_points)
t_seq = array_df[:, 2].reshape(n_cycles, n_points)

fn_total = array_df[:, 0]
h_total = array_df[:, 1] - array_df[:, 1].min()
t_total = array_df[:, 3]

# Plots configs

fonts(folder_path='C:/Users/petrus.kirsten/AppData/Local/Microsoft/Windows/Fonts/')
fig, ax1 = plt.subplots(figsize=(36 * cm, 20 * cm))

# Left axis configs
ax1.set_xlabel('Oscillation time (s)')
ax1.set_ylabel('Force (N)')
labels = ['#1 Cycle', '#1 Cycle',
          '#2 Cycle', '#2 Cycle',
          '#3 Cycle', '#3 Cycle']

for i in np.arange(0, n_cycles, 2):
    ax1.plot(np.append(t_seq[i, :], t_seq[i + 1, :] + 1),
             np.append(fn_seq[i, :], fn_seq[i + 1, :]),
             label=labels[i], color='crimson', alpha=(0.9 - 0.15 * i), lw=3.5)
ax1.legend(frameon=False)

# Right axis configs
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_xlim([0, 2])
ax2.set_ylabel('Stress (kPa)')

for i in np.arange(0, n_cycles, 2):
    ax2.plot(np.append(t_seq[i, :], t_seq[i + 1, :] + 1),
             np.append(fn_seq[i, :], fn_seq[i + 1, :]) / 0.267,
             label=labels[i], color='palevioletred', alpha=0, lw=2.5)

plt.show()
