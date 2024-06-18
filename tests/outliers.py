import numpy as np


def getCteRange(array, threshold):
    ranges = []
    init_range = 0

    for i in range(1, len(array)):
        if abs(array[i] - array[i - 1]) > threshold:
            if init_range < i - 1:
                ranges.append((init_range, i - 1))
            init_range = i

    if init_range < len(array) - 1:
        ranges.append((init_range, len(array) - 1))

    return ranges


# Exemplo de uso
dados = np.array([1, 1, 1, 2, 3, 3, 3, 2, 2, 2, 2, 5, 5, 5])
limiar_constancia = 0.5
intervalos_constantes = getCteRange(dados, limiar_constancia)
print(intervalos_constantes)
