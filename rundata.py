from __future__ import print_function
import plot
import model as m
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os

BetaValues = np.array([0.0, 0.05, 0.11, 0.13, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 2.0]) # sparse sweep
BetaValues = np.array([0.25, 0.35, 0.45, 0.55, 0.6, 0.65, 0.7])


FIGS_DIR = 'figures/'
LOGS_DIR = 'logs/'
report_loss_every_epoch = 20
beta_start_epoch   = 0
beta_rampup_epochs = 0      # Slowly phase in beta over this many epochs . 0 for no rampup
n_data = None               # consider a small subset of data (for code testing only)
n_models = 10               # simultaneously train n_models at once and save results for the best model only


def main():

    # build and train models for different values of beta
    if not os.path.exists(FIGS_DIR):
       print("Making figures directory", FIGS_DIR)
       os.mkdir(FIGS_DIR)
    if not os.path.exists(LOGS_DIR):
       print("Making logs directory", LOGS_DIR)
       os.mkdir(LOGS_DIR)


    # load training data
    data = load_mnist()

    #build_and_train_model(data, beta=0.8, save_logs=True, squared_IB_functional=False)

    # train model
    for beta in BetaValues:
        build_and_train_model(data, beta=beta, save_logs=True, squared_IB_functional=False)
    for beta in BetaValues:
        build_and_train_model(data, beta=beta, save_logs=True, squared_IB_functional=True)

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
    n_epochs = 200 #300                                              # number of epochs
    initial_lr = 0.0001                                         # initial learning rate
    learning_rate = tf.placeholder(dtype=tf.float32, shape=())  # learning rate placeholder
    beta_ph = tf.placeholder(dtype=tf.float32, shape=())        # beta placeholder

    # define placeholders
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784]) # digit images
    y = tf.placeholder(dtype=tf.float32, shape=[None, 10])  # one-hot labels

    # define models
    models = []
    for n in range(n_models):
        model_name = 'model_' + str(n)
        with tf.variable_scope(model_name):
            model = m.Model(input_ph=x, target_ph=y, learning_rate_ph=learning_rate, beta_ph=beta_ph, d=d, squared_IB_functional=squared_IB_functional, name=model_name)
        models.append(model)

    # train model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        n_mini_batches = int(len(data['train_labels']) / n_sgd)

        for epoch in range(n_epochs):
            start = time.time()

            # Only compute and save losses every report_loss_every_epoch'th epoch
            save_losses = epoch % report_loss_every_epoch == 0   

            # change beta during training
            if epoch < beta_start_epoch:
                current_beta = 0.0
            elif epoch < beta_start_epoch + beta_rampup_epochs:
                current_beta = beta * ((epoch-beta_start_epoch) / float(beta_rampup_epochs))
            else:
                current_beta = beta
            
            epoch_loss = np.zeros(len(models))
            epoch_Ixt = np.zeros(len(models))
            epoch_Iyt = np.zeros(len(models))

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

                cparams = {x: x_batch, y: y_batch, learning_rate: lr, beta_ph: current_beta}

                for i, model in enumerate(models):
                    # estimate eta (i.e., the kernel width of the GMM)
                    if batch == 0:
                        dm = sess.run(model.distance_matrix(), feed_dict={x: x_batch})
                        model.eta_optimizer.minimize(sess, feed_dict={model.distance_matrix_ph: dm})

                    #if epoch >= beta_start_epoch + beta_rampup_epochs and hasattr(model, 'sigma_optimizer'):
                    #    model.sigma_optimizer.minimize(sess, feed_dict=cparams)
                        
                    # apply gradient descent
                    sess.run(model.training_step(), feed_dict=cparams)

                    if save_losses:
                        # compute loss (for diagnostics)
                        loss = sess.run(model.loss(), feed_dict=cparams)
                        Ixt  = sess.run(model.Ixt(), feed_dict=cparams)
                        Iyt  = sess.run(model.Iyt(), feed_dict=cparams)
                        epoch_loss[i] += loss/n_mini_batches
                        epoch_Ixt[i] += Ixt/n_mini_batches
                        epoch_Iyt[i] += Iyt/n_mini_batches

                    if epoch == 0 and batch == 0:
                        model.update_learning_curves(epoch, loss, Ixt, Iyt)

            for i, model in enumerate(models):
                if save_losses:
                    model.update_learning_curves(epoch, epoch_loss[i], epoch_Ixt[i], epoch_Iyt[i])

                # plot training figure (for diagnostics)
                if save_losses:
                    T, T_no_noise = sess.run(model.encoder(), feed_dict={x: train_data[:]})  # originally used 20000 examples
                    plt.figure(i, figsize=(12, 2))
                    plt.clf()
                    plot.plot_training_figures(model.learning_curve_epochs, model.learning_curve, model.Ixt_curve, model.Iyt_curve, T, T_no_noise, train_labels[:], beta_string, model.name)
                    if save_logs:
                        figurefilename = FIGS_DIR+'training_' + file_name + '_' + model.name
                        print("* Updated ", figurefilename)
                        plt.savefig(figurefilename)

            # print output
            log_sigma2 = sess.run(models[0].log_sigma2)
            log_eta2 = sess.run(models[0].log_eta2)
            print()
            print('epoch', epoch+1, '/', n_epochs)
            print('current/final beta:', current_beta, beta)
            print('learning rate:', lr)
            print('noise variance:', np.exp(log_sigma2))
            print('kernel width:', np.exp(log_eta2))
            if save_losses:
                print('loss:', epoch_loss[0])
                print('mutual info I(X;T):', epoch_Ixt[0])
                print('mutual info I(Y;T):', epoch_Iyt[0])
            print('time:', time.time() - start)

        # save results to text files
        if save_logs:

            # only log data for the best model (i.e., the lowest loss)
            best_model = models[0]
            for model in models:
                if model.learning_curve[-1] < best_model.learning_curve[-1]:
                    best_model = model
            model = best_model

            T, T_no_noise = sess.run(model.encoder(), feed_dict={x: train_data[:]})
            plt.figure(i, figsize=(12, 2))
            plt.clf()
            plot.plot_training_figures(model.learning_curve_epochs, model.learning_curve, model.Ixt_curve, model.Iyt_curve, T, T_no_noise, train_labels[:], beta_string, model.name)
            figurefilename = FIGS_DIR+'best_training_' + file_name + '_' + model.name
            print("* Updated ", figurefilename)
            plt.savefig(figurefilename)


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
                Ixt = sess.run(model.Ixt(), feed_dict={x: data['test_data'][i]})
                Iyt = sess.run(model.Iyt(), feed_dict={x: data['test_data'][i], y: data['test_labels'][i]})

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
    train_data = np.reshape(train_data, (-1, 784)) / (255./2.) - 1.
    test_data = np.reshape(test_data, (-1, 784)) / (255./2.) - 1.
    train_labels = one_hot(train_labels)
    test_labels = one_hot(test_labels)

    if n_data is not None:
        data = {'train_data': train_data[:n_data], 'train_labels': train_labels[:n_data], 'test_data': test_data[:n_data], 'test_labels': test_labels[:n_data]}
    else:
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
