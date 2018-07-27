import numpy as np
import matplotlib.pyplot as plt
import plot
from rundata import BetaValues, LOGS_DIR, FIGS_DIR

# load data from text files and plot figures

# load data for IB curves
I_xt_squared_IB, I_yt_squared_IB, I_xt, I_yt = [], [], [], []
for beta in BetaValues:
    beta_string = '%.3f' % beta
    file_name = 'IB2_beta_' + beta_string.replace('.', '-')
    learning_curves = np.loadtxt(LOGS_DIR+'learning_curves_' + file_name + '.txt')
    I_xt_squared_IB.append(learning_curves[-1, 1])
    I_yt_squared_IB.append(learning_curves[-1, 2])

    file_name = 'IB_beta_' + beta_string.replace('.', '-')
    learning_curves = np.loadtxt(LOGS_DIR+'learning_curves_' + file_name + '.txt')
    I_xt.append(learning_curves[-1, 1])
    I_yt.append(learning_curves[-1, 2])
I_xt_squared_IB = np.array(I_xt_squared_IB)
I_yt_squared_IB = np.array(I_yt_squared_IB)
I_xt = np.array(I_xt)
I_yt = np.array(I_yt)
I_xt_test_squared_IB = np.loadtxt(LOGS_DIR+'test_set_results_IB2.txt', usecols=1)[-len(BetaValues):]
I_yt_test_squared_IB = np.loadtxt(LOGS_DIR+'test_set_results_IB2.txt', usecols=2)[-len(BetaValues):]
I_xt_test = np.loadtxt(LOGS_DIR+'test_set_results_IB_.txt', usecols=1)[-len(BetaValues):]
I_yt_test = np.loadtxt(LOGS_DIR+'test_set_results_IB_.txt', usecols=2)[-len(BetaValues):]

if False:
    # plot IB curves
    plt.figure(100, figsize=(8, 2.5))
    ixs = np.logical_and(BetaValues >= 0.0, BetaValues <= 1.0)
    plot.plot_IB_curves(I_xt[ixs], I_yt[ixs], 
                        I_xt_test[ixs], I_yt_test[ixs],
                        BetaValues[ixs])
    plt.savefig(FIGS_DIR+'IB_curves.pdf', bbox_inches='tight')

    plt.figure(101, figsize=(8, 2.5))
    plot.plot_IB_curves(I_xt_squared_IB, I_yt_squared_IB, I_xt_test_squared_IB, I_yt_test_squared_IB, BetaValues)
    plt.savefig(FIGS_DIR+'IB2_curves.pdf', bbox_inches='tight')

if False:
    # plot scatter plots
    plt.figure(102, figsize=(5, 5))
    plot.plot_scatter_plots(LOGS_DIR,BetaValues, 'IB_beta_')
    plt.savefig(FIGS_DIR+'IB_scatter.png', dpi=300)

    plt.figure(103, figsize=(5, 5))
    plot.plot_scatter_plots(LOGS_DIR,BetaValues, 'IB2_beta_')
    plt.savefig(FIGS_DIR+'IB2_scatter.png', dpi=300)

if True:
    # plot inline
    plt.figure(104, figsize=[4, 6])
    plot.plot_inline(LOGS_DIR, I_xt, I_yt, I_xt_squared_IB, I_yt_squared_IB, BetaValues)
    plt.savefig(FIGS_DIR+'IB_inline.pdf', bbox_inches='tight')

plt.show()
