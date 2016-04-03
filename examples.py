# aggregation of the tests of the different packages

from configparser import ConfigParser as ConfigParser

from utils import fio
from svm import test_svm

def main():
    config = ConfigParser()
    config.read('config.ini')
    print("Config sections: %s" % config.sections())

    print("Reading MNIST data")
    x, y = fio.parse_mnist(config.get('MNIST', 'testset'), 100)
    print("   parsed %s lines" % x.shape[0])

    print("Get a plot path")
    pp = fio.plot_file("test")
    print("   " + pp)
    test_svm.run()


    csv_data = fio.import_csv_data(config.get('MNIST', 'testset'))
    lables, data = fio.split_labels_data(csv_data, 0)
    print("Lables length: %i" % len(lables))
    print("Data length: %i" % len(data))
    sample_data = fio.get_random_data_sample(data, 100)
    print("Sample data length: %i" % len(sample_data))
    print("Sample data type: %s" % type(sample_data))


# Program entry point. Don't execute if imported.
if __name__ == '__main__':
    main()