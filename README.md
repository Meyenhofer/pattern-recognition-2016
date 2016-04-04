# pattern-recognition-2016

## Configuration
- Python Version: 3.5.1
- Use the config.ini (see [ConfigParser])
- For a good Python style guide see [Google Python Style Guide]

## External libraries
- [numpy]: Optimized and powerful N-dimensional array object
- [scipy]: Fundamental library for scientific computing
- [sklearn]: Machine Learning in Python
- [sklearn.svm]: Support vector machines (SVMs)
- [sklearn.neural_network.multilayer_perceptron]: Generic multi layer perceptron (Github pull request recently merged), see also [here](https://github.com/scikit-learn/scikit-learn/tree/master/sklearn/neural_network)
- [bob.learn.mlp]: Bob's Multi-layer Perceptron (MLP). See [here](http://www.lfd.uci.edu/~gohlke/pythonlibs/) for Windows binaries.

## Results
### MLP
![MLP Training][fig1]
###SVN
Scores for different kernels and confusion matrices
![SVN Scores][fig2]
![linear kernel confusion matrix][fig3]
![ploy 3 kernel confusion matrix][fig4]
![poly 4 kernel confusion matrix][fig5]


[ConfigParser]: https://docs.python.org/3/library/configparser.html
[Google Python Style Guide]: https://google.github.io/styleguide/pyguide.html
[numpy]: http://www.numpy.org/
[scipy]: http://www.scipy.org/
[sklearn]: http://scikit-learn.org/
[sklearn.svm]: http://scikit-learn.org/stable/modules/svm.html
[sklearn.neural_network.multilayer_perceptron]: https://github.com/scikit-learn/scikit-learn/pull/3204
[bob.learn.mlp]: https://pypi.python.org/pypi/bob.learn.mlp


[fig1]: https://raw.githubusercontent.com/dwettstein/pattern-recognition-2016/master/figs/mlp_main.png
[fig2]: https://raw.githubusercontent.com/dwettstein/pattern-recognition-2016/master/figs/SVM-score.png
[fig3]: https://raw.githubusercontent.com/dwettstein/pattern-recognition-2016/master/figs/SVM_confusion-matrix_linear.png
[fig4]: https://raw.githubusercontent.com/dwettstein/pattern-recognition-2016/master/figs/SVM_confusion-matrix_poly_3.png
[fig5]: https://raw.githubusercontent.com/dwettstein/pattern-recognition-2016/master/figs/SVM_confusion-matrix_poly_4.png
