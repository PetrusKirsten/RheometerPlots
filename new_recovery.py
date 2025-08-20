import re
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from rheology.new_plotting import OoRecovery, statisticalAnalysis, lettersTukey
from rheology.new_plotting import plotOFS, plotBars

#
# def kappa(folderPath):
#     filePath = [
#         # kC
#         folderPath + "/kC/kC-viscoelasticRecovery-1.xlsx",
#         folderPath + "/kC/kC-viscoelasticRecovery-2.xlsx",
#         folderPath + "/kC/kC-viscoelasticRecovery-3.xlsx",
#
#         # kC CL 7
#         folderPath + "/kC_CL_7/kC_CL-viscoelasticRecovery-1.xlsx",
#         folderPath + "/kC_CL_7/kC_CL-viscoelasticRecovery-2.xlsx",
#         folderPath + "/kC_CL_7/kC_CL-viscoelasticRecovery-3.xlsx",
#         folderPath + "/kC_CL_7/kC_CL-viscoelasticRecovery-4.xlsx",
#
#         # kC CL 14
#         folderPath + "/kC_CL_14/kC_CL_14-viscoelasticRecovery-1.xlsx",
#         folderPath + "/kC_CL_14/kC_CL_14-viscoelasticRecovery-2.xlsx",
#
#         # kC CL 21
#         folderPath + "/kC_CL_21/kC_CL_21-viscoelasticRecovery-1.xlsx",
#         # folderPath + "/kC_CL_21/kC_CL_21-viscoelasticRecovery-2.xlsx",
#         folderPath + "/kC_CL_21/kC_CL_21-viscoelasticRecovery-3.xlsx",
#
#         # kC CL 28
#         folderPath + "/kC_CL_28/kC_CL_28-viscoelasticRecovery-1.xlsx",
#         folderPath + "/kC_CL_28/kC_CL_28-viscoelasticRecovery-2.xlsx",
#
#         # kC CL 42
#         folderPath + "/kC_CL_42/kC_CL_42-viscoelasticRecovery-1.xlsx",
#         folderPath + "/kC_CL_42/kC_CL_42-viscoelasticRecovery-2.xlsx",
#         # folderPath + "/kC_CL_42/kC_CL_42-viscoelasticRecovery-3.xlsx",
#         # folderPath + "/kC_CL_42/kC_CL_42-viscoelasticRecovery-4.xlsx",
#     ]
#
#     kappa_keySamples = {
#         'kCar': [],
#         'kCar/CL-7': [],
#         'kCar/CL-14': [],
#         'kCar/CL-21': [],
#         'kCar/CL-28': [],
#         'kCar/CL-42': []
#     }
#     kappa_nSamples = [3, 4, 2, 3, 2, 4]
#     kappa_cSamples = ['lightsteelblue', '#A773FF', '#892F99', '#AB247B', '#E64B83', '#FF0831']
#
#     kappas = Recovery(
#         filePath, 'kappas',
#         kappa_keySamples, kappa_nSamples, kappa_cSamples)
#
#     kappas.plotGmodulus(
#         f"Elastic modulus $G'$ (Pa)", (1 * 10 ** (-2), 1 * 10 ** 5),
#         f'Frequency (Hz)', (.075, 100),
#         show=False, save=False)
#
#     kappas.plotBars(
#         [0, None, None, None],
#         significance_list=[
#             ['A', 'B', 'B', 'B', 'B', 'B',
#              'A*', 'B*', 'C', 'C', 'B*', 'B*',],
#
#             ['X', 'A', 'B', 'C', 'D', 'E',
#              'X', 'A*', 'B*', 'C*', 'D*', 'D*',],
#
#             ['X', 'A', 'B', 'C', 'D', 'E',
#              'X', 'A*', 'B*', 'C*', 'D*', 'D*', ],
#
#             ['X', 'A', 'B', 'B', 'B', 'B',
#              'X', 'A*', 'A*', 'B*', 'C*', 'C*',],
#         ],
#         show=False, save=True)
#
#
# def iota(folderPath):
#     filePath = [
#         # iC CL 7
#         folderPath + "/iC_CL_7/iC_CL_7-viscoelasticRecovery-1.xlsx",
#         folderPath + "/iC_CL_7/iC_CL_7-viscoelasticRecovery-2.xlsx",
#
#         # iC CL 14
#         folderPath + "/iC_CL_14/iC_CL_14-viscoelasticRecovery-1.xlsx",
#         folderPath + "/iC_CL_14/iC_CL_14-viscoelasticRecovery-2.xlsx",
#
#         # iC CL 21
#         folderPath + "/iC_CL_21/iC_CL_21-viscoelasticRecovery-1.xlsx",
#         folderPath + "/iC_CL_21/iC_CL_21-viscoelasticRecovery-2.xlsx",
#
#         # iC CL 28
#         folderPath + "/iC_CL_28/iC_CL_28-viscoelasticRecovery-1.xlsx",
#         folderPath + "/iC_CL_28/iC_CL_28-viscoelasticRecovery-2.xlsx",
#
#         # iC CL 42
#         folderPath + "/iC_CL_42/iC_CL_42-viscoelasticRecovery-1.xlsx",
#         folderPath + "/iC_CL_42/iC_CL_42-viscoelasticRecovery-2.xlsx",
#         folderPath + "/iC_CL_42/iC_CL_42-viscoelasticRecovery-3.xlsx",
#     ]
#
#     keySamples = {
#         'iCar/CL_7': [],
#         'iCar/CL_14': [],
#         'iCar/CL_21': [],
#         'iCar/CL_28': [],
#         'iCar/CL_42': []
#     }
#     nSamples = [2, 2, 2, 2, 3, ]
#     cSamples = ['greenyellow', '#80ed99', '#57cc99', '#38a3a5', '#22577a']
#
#     iotas = Recovery(
#         filePath, 'iota',
#         keySamples, nSamples, cSamples)
#
#     iotas.plotGmodulus(
#         f"Elastic modulus $G'$ (Pa)", (1 * 10 ** (-2), 3 * 10 ** 3),
#         f'Frequency (Hz)', (.075, 100),
#         show=False)
#
#     iotas.plotBars(
#         [None, None, None, None],
#         significance_list=[
#             ['A', 'B', 'B', 'B', 'B',
#              'A', 'A', 'B*', 'B', 'A',],
#
#             [' ', 'A', 'B', 'A', 'A',
#              ' ', 'A*', 'B*', 'B*', 'A*',],
#
#             ['A', 'B', 'C', 'B', 'B',
#              'A*', 'B*', 'C*', 'C*', 'C*',],
#
#             [' ', 'A', 'A', 'A', 'A',
#              'A*', 'B*', 'C*', 'C*', 'B*',],
#         ],
#         show=False, save=True)
#
#     # iotas.plotHeatMap(show=False)

def kappa(folderPath):

    filePath_0 = [
        # kC
        folderPath + "/kC/kC-viscoelasticRecovery-1.xlsx",
        folderPath + "/kC/kC-viscoelasticRecovery-2.xlsx",
        folderPath + "/kC/kC-viscoelasticRecovery-3.xlsx",
    ]
    filePath_7 = [
        # kC CL 7
        folderPath + "/kC_CL_7/kC_CL-viscoelasticRecovery-1.xlsx",
        folderPath + "/kC_CL_7/kC_CL-viscoelasticRecovery-2.xlsx",
        folderPath + "/kC_CL_7/kC_CL-viscoelasticRecovery-3.xlsx",
        folderPath + "/kC_CL_7/kC_CL-viscoelasticRecovery-4.xlsx",
    ]
    filePath_14 = [
        # kC CL 14
        folderPath + "/kC_CL_14/kC_CL_14-viscoelasticRecovery-1.xlsx",
        folderPath + "/kC_CL_14/kC_CL_14-viscoelasticRecovery-2.xlsx",
    ]
    filePath_21 = [
        # kC CL 21
        folderPath + "/kC_CL_21/kC_CL_21-viscoelasticRecovery-1.xlsx",
        # folderPath + "/kC_CL_21/kC_CL_21-viscoelasticRecovery-2.xlsx",
        folderPath + "/kC_CL_21/kC_CL_21-viscoelasticRecovery-3.xlsx",
    ]
    filePath_28 = [
        # kC CL 28
        folderPath + "/kC_CL_28/kC_CL_28-viscoelasticRecovery-1.xlsx",
        folderPath + "/kC_CL_28/kC_CL_28-viscoelasticRecovery-2.xlsx",
    ]
    filePath_42 = [
        # kC CL 42
        folderPath + "/kC_CL_42/kC_CL_42-viscoelasticRecovery-1.xlsx",
        folderPath + "/kC_CL_42/kC_CL_42-viscoelasticRecovery-2.xlsx",
        # folderPath + "/kC_CL_42/kC_CL_42-viscoelasticRecovery-3.xlsx",
        # folderPath + "/kC_CL_42/kC_CL_42-viscoelasticRecovery-4.xlsx",
    ]

    stCL_recovery = [
        OoRecovery(filePath_0, 'kCar CL 0', '#E64B83'),
        OoRecovery(filePath_7, 'kCar CL 7', '#E64B83'),
        OoRecovery(filePath_14, 'kCar CL 14', '#E64B83'),
        OoRecovery(filePath_21, 'kCar CL 21', '#E64B83'),
        OoRecovery(filePath_28, 'kCar CL 28', '#E64B83'),
        OoRecovery(filePath_42, 'kCar CL 42', '#E64B83')]

    # plotOFS(stCL_recovery)

    df_pre, an_pre, tk_pre = statisticalAnalysis(stCL_recovery, which='pre', param='k')
    df_post, an_post, tk_post = statisticalAnalysis(stCL_recovery, which='post', param='k')

    plotBars(
        stCL_recovery, "K", 10000,
        lettersTukey(tk_pre), lettersTukey(tk_post),
        save=True
    )
    # plotBars(stCL_recovery, 'n', .2)

def iota(folderPath):

    filePath_7 = [
        # iC CL 7
        folderPath + "/iC_CL_7/iC_CL_7-viscoelasticRecovery-1.xlsx",
        folderPath + "/iC_CL_7/iC_CL_7-viscoelasticRecovery-2.xlsx",
    ]
    filePath_14 = [
        # iC CL 14
        folderPath + "/iC_CL_14/iC_CL_14-viscoelasticRecovery-1.xlsx",
        folderPath + "/iC_CL_14/iC_CL_14-viscoelasticRecovery-2.xlsx",
    ]
    filePath_21 = [
        # iC CL 21
        folderPath + "/iC_CL_21/iC_CL_21-viscoelasticRecovery-1.xlsx",
        folderPath + "/iC_CL_21/iC_CL_21-viscoelasticRecovery-2.xlsx",
    ]
    filePath_28 = [
        # iC CL 28
        folderPath + "/iC_CL_28/iC_CL_28-viscoelasticRecovery-1.xlsx",
        folderPath + "/iC_CL_28/iC_CL_28-viscoelasticRecovery-2.xlsx",
    ]
    filePath_42 = [
        # iC CL 42
        folderPath + "/iC_CL_42/iC_CL_42-viscoelasticRecovery-1.xlsx",
        folderPath + "/iC_CL_42/iC_CL_42-viscoelasticRecovery-2.xlsx",
        folderPath + "/iC_CL_42/iC_CL_42-viscoelasticRecovery-3.xlsx",
    ]

    stCL_recovery = [
        OoRecovery(filePath_7, 'iCar CL 7', '#E64B83'),
        OoRecovery(filePath_14, 'iCar CL 14', '#E64B83'),
        OoRecovery(filePath_21, 'iCar CL 21', '#E64B83'),
        OoRecovery(filePath_28, 'iCar CL 28', '#E64B83'),
        OoRecovery(filePath_42, 'iCar CL 42', '#E64B83')]

    # plotOFS(stCL_recovery)

    df_pre, an_pre, tk_pre = statisticalAnalysis(stCL_recovery, which='pre', param='k')
    df_post, an_post, tk_post = statisticalAnalysis(stCL_recovery, which='post', param='k')

    plotBars(
        stCL_recovery, "K", 50,
        lettersTukey(tk_pre), lettersTukey(tk_post),
        save=True
    )
    # plotBars(stCL_recovery, 'n', .2)

def starch(folderPath):
    filePath = [
        # St
        folderPath + "/10St/10_0WSt-viscRec_1.xlsx",
        folderPath + "/10St/10_0WSt-viscRec_2.xlsx",

        # St/CL_7
        folderPath + "/10St_CL_7/10_0St_CL-recovery-1.xlsx",
        # folderPath + "/10St_CL_7/10_0St_CL-recovery-2.xlsx",
        folderPath + "/10St_CL_7/10_0St_CL-recovery-3.xlsx",

        # St/CL_14
        folderPath + "/10St_CL_14/St_CL_14-viscoelasticRecovery-1.xlsx",
        folderPath + "/10St_CL_14/St_CL_14-viscoelasticRecovery-2.xlsx",
        folderPath + "/10St_CL_14/St_CL_14-viscoelasticRecovery-3.xlsx",
        folderPath + "/10St_CL_14/St_CL_14-viscoelasticRecovery-4.xlsx",

        # St/CL_28
        folderPath + "/10St_CL_28/St_CL_28-viscoelasticRecovery-1.xlsx",
        folderPath + "/10St_CL_28/St_CL_28-viscoelasticRecovery-2.xlsx",
        folderPath + "/10St_CL_28/St_CL_28-viscoelasticRecovery-3.xlsx",
        folderPath + "/10St_CL_28/St_CL_28-viscoelasticRecovery-4.xlsx",
    ]

    stCL_recovery = [
        OoRecovery(filePath[:2], 'St CL 0', '#E1C96B'),
        OoRecovery(filePath[2:4], 'St CL 7', '#FFE138'),
        OoRecovery(filePath[4:8], 'St CL 14', '#F1A836'),
        OoRecovery(filePath[8:], 'St CL 21', '#E36E34')]

    # plotOFS(stCL_recovery)

    df_pre, an_pre, tk_pre = statisticalAnalysis(stCL_recovery, which='pre', param='n')
    df_post, an_post, tk_post = statisticalAnalysis(stCL_recovery, which='post', param='n')

    plotBars(
        stCL_recovery, 'n', .15,
        lettersTukey(tk_pre), lettersTukey(tk_post),
        save=True
    )
    # plotBars(stCL_recovery, 'n', .2)

def starch_kappa(folderPath):
    filePath_CL0 = [
        # St + kCar
        folderPath + "/10St_kC/10_0WSt_kCar-viscoelasticRecovery-Flow_2a.xlsx",
        folderPath + "/10St_kC/10_0WSt_kCar-viscoelasticRecovery-Flow_3a.xlsx",
        # folderPath + "/10St_kC/10_0WSt_kCar-viscoelasticRecovery-Flow_4a.xlsx",
        ]

    filePath_CL7 = [
        # St + kCar/CL_7
        folderPath + "/10St_kC_CL_7/10_0St_kC_CL-recovery-1.xlsx",
        folderPath + "/10St_kC_CL_7/10_0St_kC_CL-recovery-3_off.xlsx",
        folderPath + "/10St_kC_CL_7/10_0St_kC_CL-recovery-4.xlsx",
    ]

    filePath_CL14 = [
        # St + kCar/CL_14
        folderPath + "/10St_kC_CL_14/St_kC_CL_14-viscoelasticRecovery-1.xlsx",
        folderPath + "/10St_kC_CL_14/St_kC_CL_14-viscoelasticRecovery-2.xlsx",
        folderPath + "/10St_kC_CL_14/St_kC_CL_14-viscoelasticRecovery-3.xlsx",
        folderPath + "/10St_kC_CL_14/St_kC_CL_14-viscoelasticRecovery-4.xlsx",
    ]

    filePath_CL21 = [
        # St + kCar/CL_28
        folderPath + "/10St_kC_CL_28/St_kC_CL_28-viscoelasticRecovery-1.xlsx",
        folderPath + "/10St_kC_CL_28/St_kC_CL_28-viscoelasticRecovery-2.xlsx",
        folderPath + "/10St_kC_CL_28/St_kC_CL_28-viscoelasticRecovery-3.xlsx",
    ]

    stCL_recovery = [
        OoRecovery(filePath_CL0, 'St kCar CL 0', '#F780A7'),
        OoRecovery(filePath_CL7, 'St kCar CL 7', '#CC69B5'),
        OoRecovery(filePath_CL14, 'St kCar CL 14', '#A251C3'),
        OoRecovery(filePath_CL21, 'St kCar CL 21', '#773AD1')]

    # plotOFS(stCL_recovery)

    df_pre, an_pre, tk_pre = statisticalAnalysis(stCL_recovery, which='pre', param='n')
    df_post, an_post, tk_post = statisticalAnalysis(stCL_recovery, which='post', param='n')

    plotBars(
        stCL_recovery, 'n', .2,
        lettersTukey(tk_pre), lettersTukey(tk_post),
        save=True
    )

    # plotBars(stCL_recovery, 'n', .2)

def starch_iota(folderPath):
    filePath_CL0 = [
        # St + iCar
        folderPath + "/10St_iC/10_0WSt_iCar-viscoRecoveryandFlow_2.xlsx",
        # folderPath + "10St_iC/10_0WSt_iCar-viscoRecoveryandFlow_1.xlsx",
        folderPath + "/10St_iC/10_0WSt_iCar-viscoRecoveryandFlow_3.xlsx",
        folderPath + "/10St_iC/10_0WSt_iCar-viscoRecoveryandFlow_4.xlsx",
    ]

    filePath_CL7 = [
        # St + iCar/CL_7
        folderPath + "/10St_iC_CL_7/10_0St_iC_CL-recovery-1.xlsx",
        folderPath + "/10St_iC_CL_7/10_0St_iC_CL-recovery-2.xlsx",
        folderPath + "/10St_iC_CL_7/10_0St_iC_CL-recovery-3.xlsx",
    ]

    filePath_CL14 = [
        # St + iCar/CL_14
        folderPath + "/10St_iC_CL_14/0St_iC_CL_14-viscoelasticRecovery-1.xlsx",
        folderPath + "/10St_iC_CL_14/0St_iC_CL_14-viscoelasticRecovery-2.xlsx",
        folderPath + "/10St_iC_CL_14/0St_iC_CL_14-viscoelasticRecovery-3.xlsx",
        folderPath + "/10St_iC_CL_14/0St_iC_CL_14-viscoelasticRecovery-4.xlsx",
    ]

    filePath_CL21 = [
        # St + iCar/CL_28
        folderPath + "/10St_iC_CL_28/St_iC_CL_28-viscoelasticRecovery-1.xlsx",
        folderPath + "/10St_iC_CL_28/St_iC_CL_28-viscoelasticRecovery-2.xlsx",
        folderPath + "/10St_iC_CL_28/St_iC_CL_28-viscoelasticRecovery-3.xlsx",
    ]

    stCL_recovery = [
        OoRecovery(filePath_CL0, 'St iCar CL 0', 'lightskyblue'),
        OoRecovery(filePath_CL7, 'St iCar CL 7', '#62BDC1'),
        OoRecovery(filePath_CL14, 'St iCar CL 14', '#31A887'),
        OoRecovery(filePath_CL21, 'St iCar CL 21', '#08653A')]

    # plotOFS(stCL_recovery)

    df_pre, an_pre, tk_pre = statisticalAnalysis(stCL_recovery, which='pre', param='n')
    df_post, an_post, tk_post = statisticalAnalysis(stCL_recovery, which='post', param='n')

    # plotBars(
    #     stCL_recovery, 'K', 2500,
    #     lettersTukey(tk_pre), lettersTukey(tk_post),
    #     save=True
    # )
    plotBars(
        stCL_recovery, 'n', .2,
        lettersTukey(tk_pre), lettersTukey(tk_post),
        save=True
    )


if __name__ == '__main__':

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    # path = "C:/Users/petrus.kirsten/PycharmProjects/Rheometer-Plotting/data/by sample"  # CEBB
    # path = "C:/Users/Petrus Kirsten/Documents/GitHub/RheometerPlots/data/by sample"   # Personal
    path = "D:/Documents/GitHub/Rheometer-Plotting/data/by sample"   # New Personal

    kappa(path)
    # iota(path)

    # starch(path)
    # starch_kappa(path)
    # starch_iota(path)

    plt.show()
