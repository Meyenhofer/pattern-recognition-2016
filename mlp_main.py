from configparser import ConfigParser as ConfigParser
from utils import fio
import matplotlib.pyplot as plt
from mlp.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler


# different learning rate schedules and momentum parameters
params = [{'algorithm': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'algorithm': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'algorithm': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'algorithm': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'algorithm': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'algorithm': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'algorithm': 'adam'}]

labels = ["constant learning-rate", "constant with momentum",
          "constant with Nesterov's momentum",
          "inv-scaling learning-rate", "inv-scaling with momentum",
          "inv-scaling with Nesterov's momentum", "adam"]

plot_args = [{'c': 'red', 'linestyle': '-'},
             {'c': 'green', 'linestyle': '-'},
             {'c': 'blue', 'linestyle': '-'},
             {'c': 'red', 'linestyle': '--'},
             {'c': 'green', 'linestyle': '--'},
             {'c': 'blue', 'linestyle': '--'},
             {'c': 'black', 'linestyle': '-'}]


def plot_on_dataset(X, y, ax, name, max_iter=15):
    # for each dataset, plot learning for each learning strategy
    print("\nlearning on dataset %s" % name)
    ax.set_title(name)
    X = MinMaxScaler().fit_transform(X)
    mlps = []

    for label, param in zip(labels, params):
        print("training: %s" % label)
        mlp = MLPClassifier(verbose=0, random_state=0,
                            max_iter=max_iter, **param)
        mlp.fit(X, y)
        mlps.append(mlp)
        print("Training set score: %f" % mlp.score(X, y))
        print("Training set loss: %f" % mlp.loss_)
    for mlp, label, args in zip(mlps, labels, plot_args):
            ax.plot(mlp.loss_curve_, label=label, **args)


def main():
    config = ConfigParser()
    config.read('config.ini')
    print("Config sections: %s" % config.sections())

    # Load train set.
    csv_train_set_data = fio.import_csv_data(config.get('MNIST', 'trainingset'))
    print("CSV train data length: %i" % len(csv_train_set_data))
    #train_set_sample_data = fio.get_random_data_sample(csv_train_set_data, 2699) # Just load 10% random data while developing.
    train_set_lables, train_set_data = fio.split_labels_data(csv_train_set_data, 0)
    # Rescale.
    train_set_data = train_set_data / 255.
    print("Sample lables length: %i" % len(train_set_lables))
    print("Sample data length: %i" % len(train_set_data))
    print("Sample data type: %s" % type(train_set_data))

    # Load test set.
    csv_test_set_data = fio.import_csv_data(config.get('MNIST', 'testset'))
    print("CSV test data length: %i" % len(csv_test_set_data))
    #test_set_sample_data = fio.get_random_data_sample(csv_test_set_data, 1501) # Just load 10% random data while developing.
    test_set_lables, test_set_data = fio.split_labels_data(csv_test_set_data, 0)
    # Rescale.
    test_set_data = test_set_data / 255.


    ## mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
    ##                     algorithm='sgd', verbose=10, tol=1e-4, random_state=1)
    #mlp = MLPClassifier(hidden_layer_sizes=(1,len(train_set_data) * 0.1), max_iter=100, alpha=1e-4,
    #                    algorithm='sgd', verbose=10, tol=1e-4, random_state=1,
    #                    learning_rate_init=.1)

    #mlp.fit(train_set_data, train_set_lables)

    #print("Training set score: %f" % mlp.score(train_set_data, train_set_lables))
    #print("Test set score: %f" % mlp.score(test_set_data, test_set_lables))

    #fig, axes = plt.subplots(3, 3)
    ## use global min / max to ensure all weights are shown on the same scale
    #vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
    #for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    #    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
    #               vmax=.5 * vmax)
    #    ax.set_xticks(())
    #    ax.set_yticks(())

    #plt.show()


    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plot_on_dataset(train_set_data, train_set_lables, ax=ax, name="mnist", max_iter=15)
    fig.legend(ax.get_lines(), labels=labels, ncol=3, loc="upper center")
    plt.show()


# Program entry point. Don't execute if imported.
if __name__ == '__main__':
    main()