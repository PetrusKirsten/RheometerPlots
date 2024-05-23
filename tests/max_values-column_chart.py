import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

np.set_printoptions(threshold=np.inf)
cm = 1 / 2.54  # centimeters in inches
df = pd.read_csv("haribo_9.csv")

# df.loc[df['Fn in N'].idxmax()]
max_id = df['Fn in N'].idxmax()     # index of max force value
max_h = df['h in mm'].iloc[max_id]  # find the height where the force is max

array_df = df.drop(columns=['SegIndex']).to_numpy()
# print(array_df[:, 3])  # select values from 3rd column

n_points = 98*2
n_cycles = int(array_df[:, 3].shape[0] / n_points)

fn_seq = array_df[:, 0].reshape(n_cycles, n_points)
h_seq = array_df[:, 1].reshape(n_cycles, n_points)
t_seq = array_df[:, 3].reshape(n_cycles, n_points)

fn_total = array_df[:, 0]
h_total = array_df[:, 1]
t_total = array_df[:, 2]

# print(f'FORCE ARRAY:\n\n{fn_seq} \n\n'
#       f'HEIGHT ARRAY:\n\n{h_seq} \n\n'
#       f'SHAPE: {fn_seq.shape} \n\n'
#       f'MAX FORCE: {np.argmax(fn_seq, axis=1)}')

maxs_force_index = np.argmax(fn_seq, axis=1)

maxs_force = np.array([])
maxs_height = np.array([])

for i in range(len(maxs_force_index)):
    maxs_force = np.append(maxs_force, fn_seq[i][maxs_force_index[i]])
    maxs_height = np.append(maxs_height, h_seq[i][maxs_force_index[i]])

new_array = np.transpose(np.append(maxs_force, maxs_height).reshape(2, 3))
print(new_array)

df_fh = pd.DataFrame(new_array, columns=['Force (N)', 'Height (mm)'])
df_fh.describe()

avg = df_fh.mean().to_numpy()
std = df_fh.std().to_numpy()

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # font size of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # font size of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # font size of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # font size of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend font size
plt.rc('figure', titlesize=BIGGER_SIZE)  # font size of the figure title

plt.bar('Height of max force (mm)', avg[0], yerr=std[0], capsize=5, width=0.2)
plt.show()
