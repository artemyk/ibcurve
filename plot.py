import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import matplotlib.colors as mplc
import matplotlib.ticker as mplt

def plot_training_figures(epochs, loss, Ixt, Iyt, T, T_no_noise, labels, beta_string, model_name):

    # plot learning curves
    plt.subplot(1, 6, 1)
    plt.cla()
    plt.plot(epochs, loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(1, 6, 2)
    plt.cla()
    plt.plot(epochs, Ixt)
    plt.xlabel('epoch')
    plt.ylabel('I(X;T)')
    plt.ylim(ymin=0)

    plt.subplot(1, 6, 3)
    plt.cla()
    plt.plot(epochs, Iyt)
    plt.xlabel('epoch')
    plt.ylabel('I(Y;T)')
    plt.ylim(ymin=0, ymax=2.5)

    # plot learning trajectory
    plt.subplot(1, 6, 4)
    plt.cla()
    plt.plot([0, np.log(10)], [0, np.log(10)], 'k--')
    plt.plot(Ixt, Iyt, '.')
    plt.plot(Ixt[-1], Iyt[-1], 'y*')
    plt.plot(Ixt[0], Iyt[0], 'r.')
    plt.xlabel('I(X;T)')
    plt.ylabel('I(Y;T)')


    # plot bottleneck variables
    T_no_noise = T_no_noise - np.mean(T_no_noise, axis=0)  # center data
    T = T - np.mean(T, axis=0)

    plt.subplot(1, 6, 5)
    plt.cla()
    for label in range(10):
        ii = np.argmax(labels, axis=1) == label
        plt.scatter(T_no_noise[ii, 0], T_no_noise[ii, 1], marker='.', alpha=0.04, label=label, edgecolor='none')
    '''
    leg = plt.legend()
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    '''
    max_xy = np.max(np.abs(T_no_noise), axis=0)
    plt.xlim([-1.25 * max_xy[0], 1.25 * max_xy[0]])
    plt.ylim([-1.25 * max_xy[1], 1.25 * max_xy[1]])

    plt.subplot(1, 6, 6)
    plt.cla()
    plt.scatter(T[:, 0], T[:, 1], c=np.argmax(labels, axis=1), marker='.', alpha=0.04, edgecolor='none', cmap='tab10')
    max_xy = np.max(np.abs(T), axis=0)
    plt.xlim([-1.25 * max_xy[0], 1.25 * max_xy[0]])
    plt.ylim([-1.25 * max_xy[1], 1.25 * max_xy[1]])

    plt.suptitle('beta = ' + beta_string + ', ' + model_name, fontsize=12)
    plt.tight_layout()
    plt.pause(0.01)


def plot_IB_curves(I_xt, I_yt, I_xt_test, I_yt_test, Beta):
    markersize = 6
    
    plt.subplot(1, 3, 1)
    plt.plot([0, 4], [0, -4], '--k')
    plt.plot(I_xt, -I_yt, 'v-', markersize=markersize)
    plt.plot(I_xt_test, -I_yt_test, 'g^:', markersize=markersize)
    plt.xlabel('I(X;T)')
    plt.ylabel('-I(Y;T)')
    #plt.ylim(ymax=0)
    #plt.xlim(xmin=0)

    plt.subplot(1, 3, 2)
    plt.plot(Beta[[0, -1]], [np.log(10), np.log(10)], '--k')
    plt.plot(Beta, I_yt, 'v-', markersize=markersize)
    plt.plot(Beta, I_yt_test, 'g^:', markersize=markersize)
    plt.xlabel(r'$\beta$')
    plt.ylabel('I(Y;T)')
    #plt.ylim(ymin=0)
    plt.legend(['H(Y)', 'train', 'test'])

    plt.subplot(1, 3, 3)
    plt.plot(Beta[[0, -1]], [np.log(10), np.log(10)], '--k')
    plt.plot(Beta, I_xt, 'v-', markersize=markersize)
    plt.plot(Beta, I_xt_test, 'g^:', markersize=markersize)
    plt.xlabel(r'$\beta$')
    plt.ylabel('I(X;T)')
    #plt.ylim(ymin=0)

    plt.tight_layout()


def plot_scatter_plots(LOGS_DIR, BetaValues, file_name):
    n_beta_vals = len(BetaValues)
    for i, beta in enumerate(BetaValues):
        beta_string = '%.3f' % beta
        hidden_units = np.loadtxt(LOGS_DIR+'hidden_units_' + file_name + beta_string.replace('.', '-') + '.txt')

        labels = hidden_units[:5000, 0]
        T = hidden_units[:5000, 1:3]

        plt.subplot(np.ceil(np.sqrt(n_beta_vals)), np.ceil(np.sqrt(n_beta_vals)), i + 1)
        plt.cla()
        plt.scatter(T[:, 0], T[:, 1], c=labels, cmap='tab10', marker='.', edgecolor='none', alpha=0.1, s=2)
        max_xy = np.max(np.abs(T), axis=0)
        if i > 0:
            plt.xlim([-1.6 * max_xy[0], 1.6 * max_xy[0]])
            plt.ylim([-1.6 * max_xy[1], 1.6 * max_xy[1]])
        else:
            plt.xlim([-1.01 * max_xy[0], 1.01 * max_xy[0]])
            plt.ylim([-1.01 * max_xy[1], 1.01 * max_xy[1]])
        plt.xticks([])
        plt.yticks([])
        plt.title(r'$\beta$ = ' + beta_string, fontsize=8)

    plt.tight_layout()


def plot_inline(LOGS_DIR, I_xt_lagrangian, I_yt_lagrangian, I_xt_squared_IB, I_yt_squared_IB, Beta):
    sns.set_style('ticks')
    sns.set_context('talk')
    sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=Beta[0], vmax=Beta[-1] + 0.1))

    for p in range(2): # top plot: lagrangian, bottom plot: squared IB
        if p is 0: # lagrangian subfigure
            plt.subplot(2, 1, 1)
            I_xt = I_xt_lagrangian
            I_yt = I_yt_lagrangian
            file_name = 'IB_beta_'

            above_curve = [0, np.where(Beta == 0.4)[0], len(Beta)-1] # scatter plots above the ib curve (indices)
            below_curve = [1, np.where(Beta == 0.5)[0]]     # scatter plots below the ib curve (indices)

            plt.text(-4.5, 0.95 * 4, 'A', fontsize=20, fontweight=400)

        else: # squared IB subfigure
            plt.subplot(2, 1, 2)
            I_xt = I_xt_squared_IB
            I_yt = I_yt_squared_IB
            file_name = 'IB2_beta_'

            above_curve = [0, 5, len(Beta)-1]
            below_curve = [1, np.where(Beta == 0.5)[0]]

            plt.text(-4.5, 0.95 * 4, 'B', fontsize=20, fontweight=400)

            plt.xlabel('$I(X;T)$')

        plt.ylabel('$I(Y;T)$')
        plt.ylim([-1, 4.0])
        plt.xlim([-2, 8.1])
        plt.yticks([-1, 0, 1, 2, 3, 4])
        plt.xticks([-2, 0, 2, 4, 6, 8])
        plt.gca().tick_params(axis='both', which='major', pad=1)

        # theoretical IB curve
        max_mi_XT = np.max(I_xt)
        max_mi_YT = np.max(I_yt)
        xs = np.linspace(0, max_mi_XT, 100)
        plt.plot(xs, np.minimum(xs, max_mi_YT), color=np.array([1, 1, 1]) * 0.7, lw=6)

        # empirical results
        for i in range(len(Beta)):
            c = sm.to_rgba(Beta[i])
            plt.plot(I_xt[i], I_yt[i], '*', markersize=15, color=c)

        # scatter plots
        for i, beta in enumerate(Beta):
            scale = 0.027
            aspect_ratio = 0.57  # to make scatter plots square

            if beta == 0:
                scale /= 9

            # load data
            beta_string = '%.3f' % beta
            data = np.loadtxt(LOGS_DIR+'hidden_units_' + file_name + beta_string.replace('.', '-') + '.txt')
            labels = data[:2000, 0]
            T = data[:2000, 1:3]

            # normalize
            #T = T - np.mean(T, 0)  # np.array([x_pos, y_pos])
            T[:, 0] = T[:, 0] - np.percentile(T[:, 0], 99) + 0.5*(np.percentile(T[:, 0], 99) - np.percentile(T[:, 0], 0.01))
            T[:, 1] = T[:, 1] - np.percentile(T[:, 1], 99) + 0.5*(np.percentile(T[:, 1], 99) - np.percentile(T[:, 1], 0.01))
            T = scale * T  # / np.std(T, 0)

            # determine position
            width = 1.35
            height = width * aspect_ratio

            if i in above_curve:
                x_pos = I_xt[i] - width / 2 - 0.5
                y_pos = I_yt[i] + height / 2 + 0.25
                plt.plot([x_pos+width/2, I_xt[i]], [y_pos - height/2, I_yt[i]], 'k', lw=1)
            elif i in below_curve:
                x_pos = I_xt[i] + width / 2 + 0.5
                y_pos = I_yt[i] - height / 2 - 0.25

                if Beta[i] == 0.5 and p == 0:
                    y_pos += 0.25
                    x_pos += 0.15

                plt.plot([x_pos - width / 2, I_xt[i]], [y_pos + height / 2, I_yt[i]], 'k', lw=1)
            else:
                continue

            # plot
            plt.text(x_pos - width/2 - 0.07, y_pos + height/2 + 0.1, r'$\beta$ = %.2f' % beta, fontsize=7)
            plt.scatter(T[:, 0] + x_pos, aspect_ratio*T[:, 1] + y_pos, c=labels, cmap='tab10', marker='.', edgecolor='none', alpha=0.08, s=5)

            # draw box
            plt.gca().add_patch(
                patches.Rectangle(
                    (-width/2 + x_pos, -height/2 + y_pos),
                    width,
                    height,
                    fill=False,  # remove background
                    linewidth=1
                )
            )

    plt.plot([], label='Theoretical IB curve', color=np.array([1, 1, 1]) * 0.7, lw=6)  # proxy plots for legend
    plt.plot([], '*k', label='Empirical results', markersize=15)
    legend = plt.legend(fancybox=True, fontsize=11, frameon=True, bbox_to_anchor=(0.50, 0.35))
    legend.get_frame().set_facecolor('#EEEEFF')
    legend.get_frame().set_alpha(1.0)

    plt.tight_layout()

    #cbaxes = plt.gcf().add_axes([1.04, 0.22, 0.03, 0.7])
    #plt.colorbar(sm, label=r'$\beta$', cax=cbaxes, format='%d', ticks=[0, 1, 2])