import matplotlib ; matplotlib.use('Agg')  # Allows us to run on a headless machine
import plot
import model as m
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

FIGS_DIR = 'figures/'
LOGS_DIR = 'logs/'

def main():

    # build and train models for different values of beta

    Beta = np.append(0, np.append(10**np.linspace(start=-1.1, stop=0.2, num=20), 3))
    #Beta = np.array([0.0, 0.05, 0.11, 0.13, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 2.0]) # sparse sweep
    if True:
        # load training data
        data = load_mnist()

        # train model
        for beta in Beta:
            build_and_train_model(data, beta=beta, save_logs=True, squared_IB_functional=True)
        for beta in Beta:
            build_and_train_model(data, beta=beta, save_logs=True, squared_IB_functional=False)



    # load data from text files and plot figures
    if False:
        # load data for IB curves
        I_xt_squared_IB, I_yt_squared_IB, I_xt, I_yt = [], [], [], []
        for beta in Beta:
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
        I_xt_test_squared_IB = np.loadtxt(LOGS_DIR+'test_set_results_IB2.txt', usecols=1)[-len(Beta):]
        I_yt_test_squared_IB = np.loadtxt(LOGS_DIR+'test_set_results_IB2.txt', usecols=2)[-len(Beta):]
        I_xt_test = np.loadtxt(LOGS_DIR+'test_set_results_IB_.txt', usecols=1)[-len(Beta):]
        I_yt_test = np.loadtxt(LOGS_DIR+'test_set_results_IB_.txt', usecols=2)[-len(Beta):]

        #with open(LOGS_DIR+'train_set_results_IB2' + '.txt', 'w') as file:
        #    np.savetxt(fname=file, fmt='%.5f', X=np.array([Beta, I_xt_squared_IB, I_yt_squared_IB]).T)
        #with open(LOGS_DIR+'train_set_results_IB' + '.txt', 'w') as file:
        #    np.savetxt(fname=file, fmt='%.5f', X=np.array([Beta, I_xt, I_yt]).T)

        # plot IB curves
        plt.figure(2, figsize=(8, 3))
        #plot.plot_IB_curves(I_xt[:20], I_yt[:20], I_xt_test[:20], I_yt_test[:20], Beta[:20])
        plot.plot_IB_curves(I_xt, I_yt, I_xt_test, I_yt_test, Beta)
        #plt.savefig(FIGS_DIR+'IB_curves')

        plt.figure(3, figsize=(8, 3))
        plot.plot_IB_curves(I_xt_squared_IB, I_yt_squared_IB, I_xt_test_squared_IB, I_yt_test_squared_IB, Beta)
        #plt.savefig(FIGS_DIR+'IB2_curves')

        # plot scatter plots
        plt.figure(4, figsize=(5, 5))
        #plot.plot_scatter_plots(Beta[[0, 2, 6, 8, 13, 16, 19, 24, 25]], 'IB_beta_')
        plot.plot_scatter_plots(Beta, 'IB_beta_')
        #plt.savefig(FIGS_DIR+'IB_scatter')

        #plt.figure(5, figsize=(5, 5))
        #plot.plot_scatter_plots(Beta[[0, 2, 6, 8, 10, 12, 19, 24, 25]], 'IB2_beta_')
        #plot.plot_scatter_plots(Beta[[0, 2, 5, 6, 8, 9, 10, 21, 26]], 'IB2_beta_')
        plot.plot_scatter_plots(Beta, 'IB2_beta_')
        #plt.savefig(FIGS_DIR+'IB2_scatter')

        # plot inline
        #plt.figure(6, figsize=[4, 6])
        #plot.plot_inline(I_xt, I_yt, I_xt_squared_IB, I_yt_squared_IB, Beta)
        #plt.savefig(FIGS_DIR+'IB_inline.pdf', bbox_inches='tight')

    plt.show()


def build_and_train_model(data, beta=0.0, save_logs=False, squared_IB_functional=True):
    tf.reset_default_graph()

    # strings for file names and plots
    beta_string = '%.3f' % beta
    if squared_IB_functional:
        file_name = 'IB2_beta_' + beta_string.replace('.', '-')
    else:
        file_name = 'IB_beta_' + beta_string.replace('.', '-')

    # hyper-parameters
    d = 2                                                       # number of bottleneck hidden units
    n_sgd = 128                                                 # batch size
    n_epochs = 300                                              # number of epochs
    initial_lr = 0.0001                                         # initial learning rate
    learning_rate = tf.placeholder(dtype=tf.float32, shape=())  # learning rate placeholder

    beta_ph = tf.placeholder(dtype=tf.float32, shape=()) # not used

    # define placeholders
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784]) # digit images
    y = tf.placeholder(dtype=tf.float32, shape=[None, 10])  # one-hot labels

    # define model
    model = m.Model(input_ph=x, target_ph=y, learning_rate_ph=learning_rate, d=d, squared_IB_functional=squared_IB_functional)

    # train model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        n_mini_batches = int(len(data['train_labels']) / n_sgd)

        for epoch in range(n_epochs):
            start = time.time()

            current_beta = beta

            epoch_loss, epoch_Ixt, epoch_Iyt = 0.0, 0.0, 0.0

            # update learning rate
            #lr = initial_lr * 0.8 ** np.floor(epoch / 20)
            lr = initial_lr

            # randomize order of training data
            permutation = np.random.permutation(len(data['train_labels']))
            train_data = data['train_data'][permutation]
            train_labels = data['train_labels'][permutation]

            for batch in range(n_mini_batches):
                # sample mini-batch
                x_batch = train_data[batch * n_sgd:(1 + batch) * n_sgd]
                y_batch = train_labels[batch * n_sgd:(1 + batch) * n_sgd]

                # estimate eta (i.e., the kernel width of the GMM)
                if batch is 10:
                    dm = sess.run(model.distance_matrix(), feed_dict={x: x_batch})
                    model.eta_optimizer.minimize(sess, feed_dict={model.distance_matrix_ph: dm})

                cparams = {x: x_batch, y: y_batch, learning_rate: lr, model.beta: current_beta}
                # apply gradient descent
                sess.run(model.training_step(), feed_dict=cparams)

                # compute loss (for diagnostics)
                loss, Ixt, Iyt = sess.run(model.loss(), feed_dict=cparams)
                epoch_loss += loss/n_mini_batches
                epoch_Ixt += Ixt/n_mini_batches
                epoch_Iyt += Iyt/n_mini_batches

                if epoch is 0 and batch is 0:
                    model.update_learning_curves(loss, Ixt, Iyt)

            model.update_learning_curves(epoch_loss, epoch_Ixt, epoch_Iyt)

            # plot training figure (for diagnostics)
            if epoch % 10 is 0:
                T, T_no_noise = sess.run(model.encoder(), feed_dict={x: train_data[:20000]})
                plt.figure(1, figsize=(14, 2.5))
                plot.plot_training_figures(model.learning_curve, model.Ixt_curve, model.Iyt_curve, T, T_no_noise, train_labels[:20000], beta_string)
                if save_logs:
                    plt.savefig(FIGS_DIR+'training_' + file_name)

            log_sigma2 = sess.run(model.log_sigma2)
            log_eta2 = sess.run(model.log_eta2)

            print()
            print('epoch', epoch+1, '/', n_epochs)
            print('beta:', beta)
            print('learning rate:', lr)
            print('loss:', epoch_loss)
            print('mutual info I(X;T):', epoch_Ixt)
            print('mutual info I(Y;T):', epoch_Iyt)
            print('noise variance:', np.exp(log_sigma2))
            print('kernel width:', np.exp(log_eta2))
            print('time:', time.time() - start)

        # save results to text files
        if save_logs:
            plt.savefig(FIGS_DIR+'training_' + file_name)

            # learning curves for training data set
            with open(LOGS_DIR+'learning_curves_' + file_name + '.txt', 'w') as file:
                np.savetxt(fname=file, fmt='%.5f', X=np.array([model.learning_curve, model.Ixt_curve, model.Iyt_curve]).T)

            # scatter plots for training data
            T, T_no_noise = sess.run(model.encoder(), feed_dict={x: train_data})
            with open(LOGS_DIR+'hidden_units_' + file_name + '.txt', 'w') as file:
                np.savetxt(fname=file, fmt='%.5f', X=np.array([np.argmax(train_labels, axis=1), T[:, 0], T[:, 1], T_no_noise[:, 0], T_no_noise[:, 1]]).T)

            # final results for test data set
            Ixt_test, Iyt_test = 0, 0
            for reps in range(25):
                i = np.random.randint(low=0, high=len(data['test_data']), size=n_sgd)
                _, Ixt, Iyt = sess.run(model.loss(), feed_dict={x: data['test_data'][i], y: data['test_labels'][i], model.beta: beta})
                Ixt_test += Ixt/25
                Iyt_test += Iyt/25
            with open(LOGS_DIR+'test_set_results_' + file_name[:3] + '.txt', 'a') as file:
                file.write(file_name + ' ' + str(Ixt_test) + ' ' + str(Iyt_test) + '\n')

            print('\nsaved data logs and images\n')


def load_mnist():
    (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.mnist.load_data()

    # randomize order
    permutation = np.random.permutation(len(train_labels))
    train_data = train_data[permutation]
    train_labels = train_labels[permutation]
    permutation = np.random.permutation(len(test_labels))
    test_data = test_data[permutation]
    test_labels = test_labels[permutation]

    # normalize, reshape, and convert to one-hot vectors
    train_data = np.reshape(train_data, (-1, 784)) / (255/2) - 1
    test_data = np.reshape(test_data, (-1, 784)) / (255/2) - 1
    train_labels = one_hot(train_labels)
    test_labels = one_hot(test_labels)

    data = {'train_data': train_data, 'train_labels': train_labels, 'test_data': test_data, 'test_labels': test_labels}

    return data


def one_hot(x, n_classes=None):
    # input: 1D array of N labels, output: N x max(x)+1 array of one-hot vectors
    if n_classes is None:
        n_classes = max(x) + 1

    x_one_hot = np.zeros([len(x), n_classes])
    x_one_hot[np.arange(len(x)), x] = 1
    return x_one_hot


if __name__ == '__main__':
    main()
