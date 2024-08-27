import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Create figure and axis
fig, ax = plt.subplots()

# Set axis limits
ax.set_xlim(-5, 105)
ax.set_ylim(-5, 105)

# Draw porous scaffold background
for i in range(0, 100, 10):
    for j in range(0, 100, 10):
        circle = Circle((i, j), 4, color='lightgrey', alpha=0.6)
        ax.add_patch(circle)

# Draw cells
cell_positions = [(20, 20), (50, 50), (80, 80), (35, 70), (70, 35)]
for (x, y) in cell_positions:
    cell = Ellipse((x, y), 8, 12, color='blue', alpha=0.6)
    ax.add_patch(cell)

# Draw adhesion points on cells
for (x, y) in cell_positions:
    for angle in range(0, 360, 45):
        dx = 6 * np.cos(np.radians(angle))
        dy = 6 * np.sin(np.radians(angle))
        adhesion_point = Circle((x + dx, y + dy), 1, color='red')
        ax.add_patch(adhesion_point)

# Add text annotations for biocompatibility and bioactive factors
ax.text(50, 95, 'Biocompatible & Biodegradable Scaffold', horizontalalignment='center', fontsize=12, color='green')
ax.text(50, 85, 'High Porosity & Surface Chemistry', horizontalalignment='center', fontsize=12, color='green')
ax.text(50, 75, 'Bioactive Factors & Hydrated Environment', horizontalalignment='center', fontsize=12, color='green')

# Remove axis
ax.axis('off')

# Save and display the image
plt.savefig('cell_growth_environment.png')
plt.show()
