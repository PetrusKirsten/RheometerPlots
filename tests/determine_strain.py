import numpy as np
from matplotlib import pyplot as plt

# plt.rcParams.update({
#     "text.usetex": True
# })

y = []
x = []

for down in np.arange(0, 0.26, 0.01):
    up = down / (down + 1)
    print(f'{down * 100:.2f}%, {up * 100:.2f}%')
    # print(f'{down * 100}%, {up * 100}%')

    x.append(down * 100)
    y.append(up * 100)

plt.style.use('bmh')
plt.text(12.5, 19.5, r'$up = down / (down + 1)$', fontsize=14)

plt.xlabel('Down strain (%)')
plt.xlim([min(x), 25])
plt.ylabel('Up strain (%)')
plt.ylim([min(y), 25])

plt.plot(x, y, c='gray', lw=1.25, alpha=0.55, zorder=1)

plt.scatter(6.50, 6.10, c='tomato', s=40, marker='o', linewidths=1.5, alpha=0.85, zorder=2)
plt.text(6.5 + 0.75, 6.10 - 0.75, '(6.50%, 6.10%)', c='tomato', fontsize=11)

plt.scatter(4.00, 3.85, c='darkorange', s=40, marker='o', linewidths=1.5, alpha=0.85, zorder=2)
plt.text(4.0 + 0.75, 3.85 - 0.75, '(4.00%, 3.85%)', c='darkorange', fontsize=11)

plt.scatter(2.00, 1.96, c='dodgerblue', s=40, marker='o', linewidths=1.5, alpha=0.85, zorder=2)
plt.text(2.0 + 0.75, 1.96 - 0.75, '(2.00%, 1.96%)', c='dodgerblue', fontsize=11)

plt.show()
