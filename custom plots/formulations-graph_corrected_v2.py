import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_formulations(ax, formulations, title, carrageenan_type=None):
    ax.set_proj_type('ortho')
    ax.view_init(elev=20, azim=45)
    ax.set_xlabel('DHT time (h)')
    ax.set_ylabel('Starch concentration (%)')
    ax.set_zlabel(f'{carrageenan_type} concentration (%)' if carrageenan_type else 'Dummy Z')
    ax.set_title(title)

    for formulation in formulations:
        dht, starch, carrageenan, cacl2 = formulation
        color = 'red' if cacl2 else 'blue'
        marker = 's' if cacl2 else 'o'

        # Plot the marker
        ax.scatter(dht, starch, carrageenan, c=color, marker=marker, s=50)

        # Add drop lines to the x-y plane
        ax.plot([dht, dht], [starch, starch], [0, carrageenan], color='gray', linestyle='dashed', alpha=0.5)


# Define formulations
starch_only = [
    [0, 10, 0, False], [0, 10, 0, True],
    [1, 10, 0, False], [1, 10, 0, True],
    [2, 10, 0, False], [2, 10, 0, True],
    [0, 5, 0, False], [0, 5, 0, True],
    [1, 5, 0, False], [1, 5, 0, True],
    [2, 5, 0, False], [2, 5, 0, True]
]

iota_carrageenan = [
    [0, 10, 1, False], [0, 10, 1, True],
    [1, 10, 1, False], [1, 10, 1, True],
    [2, 10, 1, False], [2, 10, 1, True],
    [0, 5, 0.5, False], [0, 5, 0.5, True],
    [1, 5, 0.5, False], [1, 5, 0.5, True],
    [2, 5, 0.5, False], [2, 5, 0.5, True],
    [0, 0., 1., False], [0., 0., 1., True]
]

kappa_carrageenan = [
    [0., 10., 1., False], [0., 10., 1., True],
    [1., 10., 1., False], [1., 10., 1., True],
    [2., 10., 1., False], [2., 10., 1., True],
    [0., 5., .5, False], [0., .5, .5, True],
    [1., .5, .5, False], [1., .5, .5, True],
    [2., .5, .5, False], [2., .5, .5, True],
    [0., .5, .5, False], [0., .5, .5, True]
]

# Create figure and subplots
fig = plt.figure(figsize=(15 , 6))
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

# Plot formulations
plot_formulations(ax1, starch_only, 'Starch-only Formulations')
plot_formulations(ax2, iota_carrageenan, 'Iota Carrageenan Formulations', 'Iota Carrageenan')
plot_formulations(ax3, kappa_carrageenan, 'Kappa Carrageenan Formulations', 'Kappa Carrageenan')

# Adjust layout and display
plt.tight_layout()
plt.show()

# Save the figure
fig.savefig('HydrogelsFormulation.png', dpi=300, bbox_inches='tight')