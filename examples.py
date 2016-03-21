# aggregation of the tests of the different packages

from ConfigParser import ConfigParser

from utils.fio import parse_mnist, plot_file
from svm import test_svm


config = ConfigParser()
config.read('config.ini')
print "Config sections: %s" % config.sections()

print "Reading MNIST data"
x, y = parse_mnist(config.get('MNIST', 'testset'), 100)
print "   parsed %s lines" % x.shape[0]

print "Get a plot path"
pp = plot_file("test")
print "   " + pp
test_svm.run()
