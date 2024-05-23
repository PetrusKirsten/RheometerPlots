import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.font_manager import FontProperties


def fit_par(name, parameters):
    s_pt = (t_total[-1] - t_total[0]) / len(t_total)

    if len(parameters) > 4:
        print(f'\n*** {name} FITTING PARAMETERS: ***\n\n'
              f'- Amplitude: {abs(parameters[0]):.2f} N.\n'
              f'- Damping Coef.: {parameters[1]:.2f}\n'
              f'- Angular frequency: {parameters[2]:.2f} rad/pts = {parameters[2] / s_pt:.1f} rad/s.\n'
              f'- Frequency: {(parameters[2] / s_pt) / (2 * np.pi):.2f} Hz.\n'
              f'- Phase: {parameters[3] * (180 / np.pi):.1f} degrees.\n'
              f'- Displacement in y axis: {parameters[4]:.2f} N.\n')

    elif len(parameters) <= 4:
        print(f'\n*** {name} FITTING PARAMETERS: ***\n\n'
              f'- Amplitude: {parameters[0]:.2f} N.\n'
              f'- Damping Coef.: {parameters[1]:.2f}\n'
              f'- Angular frequency: {parameters[2]:.2f} rad/pts = {parameters[2] / s_pt:.1f} rad/s.\n'
              f'- Frequency: {(parameters[2] / s_pt) / (2 * np.pi):.2f} Hz.\n')



# Data config
df = pd.read_csv('../data/haribo-9v2.csv')
# print(df.to_string())

array_df = df.drop(columns=['SegIndex']).to_numpy()

n_points = 98
n_cycles = int(array_df[:, 3].shape[0] / n_points)

fn_seq = array_df[:, 0].reshape(n_cycles, n_points)
h_seq = array_df[:, 1].reshape(n_cycles, n_points)
t_seq = array_df[:, 3].reshape(n_cycles, n_points)

fn_total = array_df[:, 0]
h_total = array_df[:, 1] - array_df[:, 1].min()
t_total = array_df[:, 2]


# Curve fitting
def damped_sinusoid(t, a, lam, w, phi, y):
    return a * np.exp(lam * t) * np.sin(w * t + phi) + y


def abs_damped_sinusoid(t, a, lam, w, phi):
    return abs(a * np.exp(lam * t) * np.sin(w * t + phi))


def sinusoid(t, a, w, phi, y):
    return a * np.sin(w * t + phi) + y


popt_damped, pcov_damped = curve_fit(
    damped_sinusoid,
    t_total[45:],
    fn_total[45:],
    p0=(1.5, 0.01, 2.9, 1.75, 1.2)
)
popt_abs, pcov_abs = curve_fit(
    abs_damped_sinusoid,
    t_total[45:],
    fn_total[45:],
    p0=(3, 0.01, 1.75, 1.75)
)
popt_sin, pcov_sin = curve_fit(
    sinusoid,
    t_total,
    h_total,
    p0=(1.25, 2.9, 1.75, 1.2)
)

fit_par('DAMPED SINUSOID', popt_damped)
fit_par('ABS DAMPED SINUSOID', popt_abs)

# Plots configs
fonts(folder_path='C:/Users/petrus.kirsten/AppData/Local/Microsoft/Windows/Fonts/')
fig, ax1 = plt.subplots(figsize=(36 * cm, 20 * cm))
fig.suptitle('Dynamic oscillation in Haribo ("haribo_9.csv")', alpha=0.8)

# Left axis configs
colorax1 = 'orangered'
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Force (N)', color=colorax1)
ax1.tick_params(axis='y', labelcolor=colorax1, colors=colorax1)

ax1.scatter(
    t_total, fn_total,
    color=colorax1, alpha=0.5, s=35, edgecolors='none')  # Experimental data
ax1.plot(
    t_total[45:],
    damped_sinusoid(t_total[45:], popt_damped[0], popt_damped[1], popt_damped[2], popt_damped[3], popt_damped[4]),
    color=colorax1, alpha=0.85, label=f'Damped sinusoid - Damping coef.: {popt_damped[1]:.2f}')  # Fitted curve

ax1.plot(
    t_total[45:],
    abs_damped_sinusoid(t_total[45:], popt_abs[0], popt_abs[1], popt_abs[2], popt_abs[3]),
    color=colorax1, alpha=0.85, label=f'Abs damped sinusoid - Damping coef.: {popt_abs[1]:.2f}', ls='--')

# Right axis configs
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
colorax2 = 'dodgerblue'
ax2.spines['left'].set_color(colorax1)
ax2.spines['right'].set_color(colorax2)
ax2.set_xlim([0, t_total[-1]])
ax2.set_ylabel('Height (mm)', color=colorax2)
ax2.tick_params(axis='y', labelcolor=colorax2, colors=colorax2)

ax2.scatter(
    t_total, h_total,
    color=colorax2, alpha=0.5, s=35, edgecolors='none')
ax2.plot(
    t_total,
    sinusoid(t_total, popt_sin[0], popt_sin[1], popt_sin[2], popt_sin[3]),
    color=colorax2, alpha=0.85, label=f'Fitted damped sinusoid')  # Fitted curve

fig.tight_layout()  # otherwise the right y-label is slightly clipped
ax1.legend(frameon=False)
# ax1.text(13.7 * cm, 7.7 * cm, 'Damping coef.: -0.02', )  # Show the damping coef in chart
plt.show()
