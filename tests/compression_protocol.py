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


def sine(A=1, f=1, n=1000, phi=0):
    """
    Gera uma onda senoidal.

    Parâmetros:
    A -- Amplitude da senoidal (padrão = 1)
    f -- Frequência da senoidal (padrão = 1)
    n -- Número de pontos na onda (padrão = 100)
    phi -- Deslocamento de fase (em radianos) (padrão = 0)

    Retorna:
    x -- Array de valores no eixo x
    y -- Array de valores da função senoidal correspondentes a cada x
    """
    x = np.linspace(0, 2 * np.pi, n)  # Gera n pontos entre 0 e 2π
    y = A * np.sin(f * x + phi)  # Equação da senoidal: y = A * sin(f * x + φ)
    return x, y


def elastic_break_behavior(k=1, x_max=3, x_plastic=1, x_break=2, n=100):
    """
    Gera uma curva suavizada de força versus deformação para um material elástico até a quebra.

    Parâmetros:
    k -- Constante elástica (rigidez) do material (padrão = 1)
    x_max -- Valor máximo de deformação (padrão = 10)
    x_plastic -- Deformação onde começa o comportamento plástico (padrão = 6)
    x_break -- Deformação no ponto de quebra do material (padrão = 8)
    n -- Número de pontos na curva (padrão = 200)

    Retorna:
    x -- Array de deformações
    F -- Array de forças correspondentes
    """
    # Definir o array de deformações com n pontos
    x = np.linspace(0, x_max, n)

    # Inicializar a curva de força
    F = np.zeros_like(x)

    # Fase Elástica: Força aumenta linearmente até o limite plástico
    elastic_limit = x_plastic
    F[x <= elastic_limit] = 2 * k * x[x <= elastic_limit]

    # Suavização na transição para o platô plástico
    transition_start = elastic_limit
    transition_end = x_break
    transition_mask = (x > transition_start) & (x <= transition_end)

    # Força na fase plástica com suavização
    F[transition_mask] = (k * elastic_limit) * (1 - np.exp(-5 * (x[transition_mask] - elastic_limit))) + 2

    # Fase de Quebra: A força decai suavemente após a deformação de quebra
    break_mask = x > transition_end
    F[break_mask] = (k * elastic_limit) * np.exp(-2 * (x[break_mask] - transition_end))

    return x, F


def downsample_array(arr, x):
    """
    Reduz o tamanho do array original para n/x valores, mantendo intervalos constantes.

    Parâmetros:
    arr -- Array de entrada (1D)
    x -- Fator de downsampling (deve ser um inteiro > 0)

    Retorna:
    new_arr -- Novo array com n/x valores
    """
    if x <= 0:
        raise ValueError("O fator de downsampling deve ser um inteiro positivo.")

    n = len(arr)
    if n < x:
        raise ValueError("O array deve ter pelo menos x elementos para o downsampling.")

    # Criar um novo array com valores selecionados
    new_length = n // x
    new_arr = np.zeros(new_length)

    # Preencher o novo array com valores do array original
    for i in range(new_length):
        new_arr[i] = arr[i * x]  # Seleciona o valor correspondente

    return new_arr


fonts(folder_path='C:/Users/petrus.kirsten/AppData/Local/Microsoft/Windows/Fonts/')
# Plots configs
plt.style.use('seaborn-v0_8-ticks')
fig, axes = plt.subplots(figsize=(14, 7), facecolor='w', ncols=1)
fig.suptitle('Mechanical assay protocol to evaluate loading behaviours.')
markerSize = 30

# data
freq = 6
points = 1000
xForce, yForce = sine(A=1, f=freq, n=points, phi=0)
xForce, yForce = downsample_array(xForce, 10), downsample_array(yForce, 10)
xHeight, yHeight = sine(A=0.5, f=freq, n=points, phi=0)

xForceBreak, yForceBreak = elastic_break_behavior(k=1, x_max=3, x_plastic=1, x_break=2, n=100)
xForceBreak, yForceBreak = downsample_array(xForceBreak, 5), downsample_array(yForceBreak, 5)
xHeightBreak, yHeightBreak = elastic_break_behavior(k=0.5, x_max=3, x_plastic=4, x_break=4, n=100)
xHeightBreak += xHeight[-1]
xF, yF = np.concatenate((xForce, xForceBreak + xForce[-1])), np.concatenate((yForce, yForceBreak))
xH, yH = np.concatenate((xHeight, xHeightBreak)), np.concatenate((yHeight, yHeightBreak))

# 1st plot configs
ax1 = axes
ax1.spines[['top', 'bottom', 'left', 'right']].set_linewidth(0.75)
ax1.set_xlabel('Time')
ax1.set_xticks([xHeight[-1]])
ax1.set_xticklabels(['60 s'], size=9.5)
ax1.set_xlim([0, 9])
ax1.set_yticks([-1/2, 1/2, yF.max()])
ax1.set_yticklabels(['0%', '7%', 'Break'], size=9.5)
ax1.set_ylim([-1.1, 3.5])
ax1.grid(ls='--')

ax1.text(xHeight[xHeight.shape[0]//2], yHeightBreak.max()+0.14,
         s='Oscillatory compression to determine dynamic modulus'
           '\nwith 7% amplitude at 0.5 Hz for 60 s (n = 30).',
         horizontalalignment='center', verticalalignment='bottom',
         color='k', size=10)
ax1.vlines(xHeight[-1], -10, 10,
           lw=1, color='orange')
ax1.text(xHeightBreak[xHeightBreak.shape[0]//2]-0.13, yHeightBreak.max()+0.14,
         s='Compression until breakage to determine'
         '\ncompressive modulus at 1 mm/s.',
         horizontalalignment='center', verticalalignment='bottom',
         color='k', size=10)

ax1.scatter(xF, yF,
            color='deeppink', edgecolor='k', lw=.5, s=markerSize, alpha=0.75,
            label='F', zorder=2)
ax1.plot(xH, yH,
         color='lightskyblue', lw=1.2, alpha=0.8,
         label='Height', zorder=1)

ax1.legend(loc=2, frameon=False)
plt.subplots_adjust(wspace=0.0, top=0.93, bottom=0.08, left=0.05, right=0.95)
plt.show()
fig.savefig('compression_protocol.png', dpi=600)
