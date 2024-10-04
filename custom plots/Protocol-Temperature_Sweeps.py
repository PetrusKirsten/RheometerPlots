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
fig, axes = plt.subplots(figsize=(8, 7), facecolor='w', ncols=1)
fig.suptitle('Rheometry assay protocol to evaluate thermal stability.\n\n')
markerSize = 50

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

# 1st plot configs
ax1 = axes
ax1.set_title('Temperature sweep assay.',
              size=10)
ax1.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)
ax1.set_xlabel('Time')
ax1.set_xticks([])
ax1.set_xlim([-0.5, 10.5])
ax1.set_yticks([1, 10])
ax1.tick_params(axis='y', colors='crimson')
ax1.set_yticklabels(['$5\,^oC$', '$50\,^oC$'], size=9.5, color='crimson')
ax1.set_ylim([0.5, 10.5])
ax1.grid(ls='--', color='crimson', alpha=0.5)

ax1.plot(xFreq, yFreq,
         color='crimson', lw=1.5, alpha=0.8,
         label='T', zorder=1)
ax1.scatter(xMod, yStoMod,
            lw=.5, c='slategrey', edgecolor='k', s=markerSize, marker='v',
            label="G'", zorder=2)
ax1.scatter(xMod, yLossMod,
            lw=.5, c='lightsteelblue', edgecolor='k', s=markerSize, marker='^',
            label='G"', zorder=2)
ax1.legend(frameon=True)
ax1.legend(frameon=True).get_frame().set_facecolor('whitesmoke')
ax1.legend(frameon=True).get_frame().set_edgecolor('whitesmoke')
# Draw rectangles for resting and setting temperature time
# textCoord = (-1.5, 10.1)
# textColor = 'k'
# textLabel = 'Rest and\nSet $T=37\,^oC$\nfor $180\,\,s$'
# textSize = 9.2
# ax1.text(textCoord[0], textCoord[1], s=textLabel,
#          horizontalalignment='center', verticalalignment='top',
#          color=textColor, size=textSize)
# rect = Rectangle((-3, 0), 3, 10.5,
#                  linewidth=1, edgecolor='k', facecolor='whitesmoke', alpha=0.75,
#                  zorder=0)
# ax1.add_patch(rect)

plt.subplots_adjust(wspace=0.0, top=0.91, bottom=0.05, left=0.09, right=0.95)
plt.show()
filename = 'temperature_protocol'
fig.savefig(filename + '.png', facecolor='w', dpi=600)
