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
plt.xlabel('Down strain (%)')
plt.xlim([min(x), max(x)])
plt.ylabel('Up strain (%)')
plt.ylim([min(y), max(y)])
plt.plot(x, y, c='crimson')
plt.scatter(6.5, 6.1, c='crimson', s=50)
plt.text(6.5, 6.1 - 1.2, '(6.50%, 6.10%)', fontsize=11)
plt.scatter(4, 3.85, c='crimson', s=50)
plt.text(4, 3.85 - 1.2, '(4.00%, 3.85%)', fontsize=11)
plt.text(2.5, 18.5, r'$up = down / (down + 1)$', fontsize=14)
plt.show()
