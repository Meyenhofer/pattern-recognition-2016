from utils import fio


def main():
    """
    collection of examples to use the different modules and functions
    """

    # Get the configurations
    config = fio.get_config()
    print('Config sections: %s' % config.sections())

    # Read text data
    print('Reading MNIST data')
    x, y = fio.parse_mnist(config.get('MNIST', 'testset'), 100)
    print('   parsed %s lines' % x.shape[0])

    print('Read csv data')
    csv_data = fio.import_csv_data(config.get('MNIST', 'testset'))
    labels, data = fio.split_labels_data(csv_data, 0)
    print('   Labels length: %i' % len(labels))
    print('   Data length: %i' % len(data))
    sample_data = fio.get_random_data_sample(data, 100)
    print('   Sample data length: %i' % len(sample_data))
    print('   Sample data type: %s' % type(sample_data))

    # Get a path for a (internal) plot file
    print('Get a plot path')
    pp = fio.get_plot_file('test')
    print('   ' + pp)


# Program entry point. Don't execute if imported.
if __name__ == '__main__':
    main()
