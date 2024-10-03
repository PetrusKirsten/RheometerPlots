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


def thixo_exp(n=20, b=2, p=2.5, d=.5):
    """
    Gera uma série de 'n' pontos com comportamento tixotrópico e decaimento exponencial.

    Parâmetros:
    n -- número de pontos (padrão é 10)
    b -- valor base da série (inicial e final)
    p -- valor máximo no ápice
    d -- taxa de decaimento exponencial para retornar ao valor base
    """
    # Valores iniciais constantes no nível base
    initial = np.full(0, b)
    # Crescimento linear até o valor máximo
    growth = np.linspace(b, p, 1)
    # Decaimento exponencial para retornar ao valor base
    decay_length = n - len(initial) - len(growth)
    decay = p * np.exp(-d * np.arange(decay_length))

    # Ajustar decaimento para retornar ao valor base suavemente
    final_decay = b + (decay - b)

    # Combinar segmentos
    series = np.concatenate([growth, final_decay])

    return series


def sigma(g, k=1, n=0.5):
    """
    Calcula a tensão de cisalhamento (τ) para um material pseudoplástico.

    Parâmetros:
    g -- array ou lista de taxas de cisalhamento (shear rate)
    k -- consistência do fluido (padrão é 1)
    n -- índice de comportamento de fluxo (padrão é 0.5 para pseudoplástico)

    Retorna:
    τ -- Tensão de cisalhamento calculada para cada taxa de cisalhamento
    """
    # Aplicar a lei de potência: τ = k * g^n
    return k * (g ** n)


fonts(folder_path='C:/Users/petrus.kirsten/AppData/Local/Microsoft/Windows/Fonts/')
# Plots configs
plt.style.use('seaborn-v0_8-ticks')
fig, axes = plt.subplots(figsize=(16, 16 // 3), facecolor='w', ncols=3)
fig.suptitle('Rheometry assay protocol to evaluate viscoelastic recovery.\n\n\n')
markerSize = 50

# 1st plot configs
ax1 = axes[0]
ax1.set_title('Oscillatory frequency sweep assay.',
              size=10)
ax1.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)
ax1.set_xticks([])
ax1.set_xlim([-3, 10.5])
ax1.set_yticks([1, 10])
ax1.tick_params(axis='y', colors='dodgerblue')
ax1.set_yticklabels(['$0.1\,Hz$', '$100\,Hz$'], size=9.5, color='dodgerblue')
ax1.set_ylim([0, 10.5])
ax1.grid(ls='--', color='dodgerblue', alpha=0.5, zorder=0)

xFreq = np.linspace(0, 10, 1000)
yFreq = np.ceil(xFreq)  # np.ceil creates a step effect by rounding up

# G' data config
n_points = 10
base_value = 8
xMod = np.linspace(0, 10, n_points + 1)
initial_points = base_value + np.random.uniform(-0.1, 0.1, n_points - 3)
yStoMod = np.concatenate([
    [base_value, base_value],  # Os primeiros 2 pontos paralelos ao eixo X
    initial_points,  # Pontos ligeiramente variáveis
    [base_value - 0.5, base_value - 1]  # Últimos pontos suavemente decrescendo
])
base_value = base_value / 1.5
initial_points = base_value + np.random.uniform(-0.1, 0.1, n_points - 3)
yLossMod = np.concatenate([
    [base_value, base_value],  # Os primeiros 2 pontos paralelos ao eixo X
    initial_points,  # Pontos ligeiramente variáveis
    [base_value + 0.5, base_value + 1]  # Últimos pontos suavemente decrescendo
])

ax1.scatter(0, 0,
            c='w', edgecolor='w', s=0, marker='o', alpha=0,
            label='σ = cte.', zorder=0)
ax1.plot(xFreq, yFreq,
         color='dodgerblue', lw=1.25, alpha=0.85,
         label='ω', zorder=1)
ax1.scatter(xMod, yStoMod,
            lw=.5, c='aquamarine', edgecolor='k', s=markerSize, marker='v',
            label="G'", zorder=2)
ax1.scatter(xMod, yLossMod,
            lw=.5, c='mediumaquamarine', edgecolor='k', s=markerSize, marker='^',
            label='G"', zorder=2)
# Draw rectangles for resting and setting temperature time
textCoord = (-1.5, 10.1)
textColor = 'k'
textLabel = 'Rest and\nSet $T=37\,^oC$\nfor $180\,\,s$'
textSize = 9.2
ax1.text(textCoord[0], textCoord[1], s=textLabel,
         horizontalalignment='center', verticalalignment='top',
         color=textColor, size=textSize)
rect = Rectangle((-3, 0), 3, 10.5,
                 linewidth=1, edgecolor='k', facecolor='whitesmoke', alpha=0.75,
                 zorder=1)
ax1.add_patch(rect)

ax1.legend(loc=4, frameon=True)
ax1.legend(loc=4, frameon=True).get_frame().set_facecolor('whitesmoke')
ax1.legend(loc=4, frameon=True).get_frame().set_edgecolor('whitesmoke')

# 2nd plot configs
ax2 = axes[1]
ax2.set_title('Flow shearing assay.',
              size=10)
ax2.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)
ax2.set_xticks([])
ax2.set_xlim([-10, 10])
ax2.set_xlabel('Time')
ax2.set_yticks([])
ax2.set_ylim([-0.15, 3.3])

yShearScte = thixo_exp()
xShearScte = np.linspace(0, 10, yShearScte.shape[0])
xShearR = np.linspace(0, 10, 1000)
yShearR = np.ceil(xShearR) / 3.5  # np.ceil creates a step effect by rounding up

xShearRcte = np.linspace(-10, 0, 20)
yShearRcte = np.full(20, yShearR[-1])

xShearS = np.linspace(xShearR[0], 10, 20)
yShearS = sigma(xShearS, k=0.65, n=0.5)

ax2.text(xShearRcte[xShearRcte.shape[0]//2], 2.9,
         s='Constant strain rate'
           '\n$\dot{γ}=100 \,\,\, s^{-1}$',
         horizontalalignment='center', verticalalignment='bottom',
         color='k', size=textSize)
ax2.text(xShearR[xShearR.shape[0]//2], 2.9,
         s='Decreasing strain rate.'
         '\n$100 > \dot{γ} > 0.1 \,\,\, s^{-1}$',
         horizontalalignment='center', verticalalignment='bottom',
         color='k', size=textSize)
ax2.vlines(-xShearR[-1] + 10, -10, 10,
           lw=1, color='slategrey')

ax2.plot(xShearRcte, yShearRcte,
         color='orange', lw=1.25, alpha=0.75,
         zorder=2)
ax2.plot(-xShearR + 10, yShearR,
         color='orange', lw=1.25, alpha=0.75,
         label='$\dot{γ}$', zorder=2)
ax2.scatter(xShearScte - 10, yShearScte,
            lw=.5, c='hotpink', edgecolor='k', s=markerSize - 20, marker='o',
            label='σ', zorder=3)
ax2.scatter(xShearS, yShearS,
            lw=.5, c='hotpink', edgecolor='k', s=markerSize - 20, marker='o',
            zorder=3)
ax2.legend(loc=4, frameon=True)
ax2.legend(loc=4, frameon=True).get_frame().set_facecolor('whitesmoke')
ax2.legend(loc=4, frameon=True).get_frame().set_edgecolor('whitesmoke')

# 3rd plot configs
ax3 = axes[2]
ax3.set_title('Oscillatory frequency sweep assay again.',
              size=10)
ax3.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)
ax3.set_xticks([])
ax3.set_xlim([-3, 10.5])
ax3.set_yticks([1, 10])
ax3.tick_params(axis='y', colors='dodgerblue')
ax3.set_yticklabels(['$0.1\,Hz$', '$100\,Hz$'], size=9.5, color='dodgerblue')
ax3.set_ylim([0, 10.5])
ax3.yaxis.tick_right()
ax3.yaxis.set_label_position('right')
ax3.grid(ls='--', color='dodgerblue', alpha=0.5, zorder=0)

base_value = 6.5
xMod = np.linspace(0, 10, n_points + 1)
initial_points = base_value + np.random.uniform(-0.2, 0.1, n_points - 3)
yStoMod = np.concatenate([
    [base_value, base_value],  # Os primeiros 2 pontos paralelos ao eixo X
    initial_points,  # Pontos ligeiramente variáveis
    [base_value - 0.5, base_value - 1]  # Últimos pontos suavemente decrescendo
])
base_value = base_value / 1.2
initial_points = base_value + np.random.uniform(-0.2, 0.1, n_points - 3)
yLossMod = np.concatenate([
    [base_value, base_value],  # Os primeiros 2 pontos paralelos ao eixo X
    initial_points,  # Pontos ligeiramente variáveis
    [base_value + 0.5, base_value + 1]  # Últimos pontos suavemente decrescendo
])

ax3.scatter(0, 0,
            c='w', edgecolor='w', s=0, marker='o', alpha=0,
            label='σ = cte.', zorder=0)
ax3.plot(xFreq, yFreq,
         color='dodgerblue', lw=1.25, alpha=0.85,
         label='ω', zorder=1)
ax3.scatter(xMod, yStoMod,
            lw=.5, c='aquamarine', edgecolor='k', s=markerSize, marker='v',
            label="G'", zorder=2)
ax3.scatter(xMod, yLossMod,
            lw=.5, c='mediumaquamarine', edgecolor='k', s=markerSize, marker='^',
            label='G"', zorder=2)
ax3.legend(loc=4, frameon=True)
ax3.legend(loc=4, frameon=True).get_frame().set_facecolor('whitesmoke')
ax3.legend(loc=4, frameon=True).get_frame().set_edgecolor('whitesmoke')

# Draw rectangles for resting and setting temperature time
ax3.text(textCoord[0], textCoord[1], s=textLabel,
         horizontalalignment='center', verticalalignment='top',
         color=textColor, size=textSize)
rect = Rectangle((-3, 0), 3, 10.5,
                 linewidth=1, edgecolor='k', facecolor='whitesmoke', alpha=0.75,
                 zorder=1)
ax3.add_patch(rect)

plt.subplots_adjust(wspace=0.0, top=0.875, bottom=0.05, left=0.05, right=0.95)
plt.show()
filename = 'viscelas-recovery_protocol'
fig.savefig(filename + '.png', facecolor='w', dpi=300)
