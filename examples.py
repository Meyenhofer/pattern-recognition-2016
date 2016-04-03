from utils import fio


def run():
    """
    This is a collection of small examples on how to use the different modules etc.
    """

    # Configs
    config = fio.get_config()
    print('Config sections: %s' % config.sections())

    # read data
    print('Reading MNIST data:')
    x, y = fio.parse_mnist(config.get('MNIST', 'testset'), 100)
    print('   parsed %s lines' % x.shape[0])

    print("CSV reading:")
    labels, data = fio.import_csv_data(config.get('MNIST', 'testset'))
    print('   Labels length: %i' % len(labels))
    print('   Data length: %i' % len(data))
    sample_data = fio.get_random_data_sample(data, 100)
    print('   Sample data length: %i' % len(sample_data))
    print('   Sample data type: %s' % type(sample_data))

    # Generate a path for a plot file
    pp = fio.get_plot_file('test')
    print('Get a path for a plot file:')
    print('   %s' % pp)


if __name__ == '__main__':
    run()
