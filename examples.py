# aggregation of the tests of the different packages

from configparser import ConfigParser as ConfigParser

from utils import fio
from svm import test_svm


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


labels, data = fio.import_csv_data(config.get('MNIST', 'testset'))
print("Labels length: %i" % len(labels))
print("Data length: %i" % len(data))
sample_data = fio.get_random_data_sample(data, 100)
print("Sample data length: %i" % len(sample_data))
print("Sample data type: %s" % type(sample_data))
