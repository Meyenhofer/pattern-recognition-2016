from utils import fio
from svm import test_svm
from mlp import mlp_main
from utils.transcription import get_transcription


def main():
    """
    collection of examples to use the different modules and functions
    """

    # Get the configurations
    config = fio.get_config()
    print('Config sections: %s\n' % config.sections())

    # Read KWS transcription
    trans = get_transcription()
    c = trans[0][0]
    s = trans[0][1]
    print('Transcription code:\n\tid: %s\tdoc-id: %s\tline-id: %s\tword-id:%s\n\tcode: %s\tstring: %s\n' %
          (c, c.get_doc(), c.get_line(), c.get_word(), s.get_word_code(), s))

    # Read text data
    print('Reading MNIST data')
    x, y = fio.parse_mnist(config.get('MNIST', 'testset'), 100)
    print('   parsed %s lines\n' % x.shape[0])

    print('Read csv data')
    csv_data = fio.import_csv_data(config.get('MNIST', 'testset'))
    labels, data = fio.split_labels_data(csv_data, 0)
    print('   Labels length: %i' % len(labels))
    print('   Data length: %i' % len(data))
    sample_data = fio.get_random_data_sample(data, 100)
    print('   Sample data length: %i' % len(sample_data))
    print('   Sample data type: %s\n' % type(sample_data))

    # Get a path for a (internal) plot file
    pp = fio.get_plot_file('test')
    print('Get a plot path:\n\t%s\n' % pp)

    # SVM test
    test_svm.run()

    # MLP test
    mlp_main.main()


# Program entry point. Don't execute if imported.
if __name__ == '__main__':
    main()
