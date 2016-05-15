# pattern-recognition-2016

## Configuration
- Python Version: 3.5.1
- Use the config.ini (see [ConfigParser])
- For a good Python style guide see [Google Python Style Guide]

## External libraries
- [cython]: cython
- [numpy]: Optimized and powerful N-dimensional array object
- [scipy]: Fundamental library for scientific computing
- [scikit-image]: Collection of algorithms for image processing
- [sklearn]: Machine Learning in Python
- [sklearn.svm]: Support vector machines (SVMs)
- [sklearn.neural_network.multilayer_perceptron]: Generic multi layer perceptron (Github pull request recently merged), see also [here](https://github.com/scikit-learn/scikit-learn/tree/master/sklearn/neural_network)
- [svg.path]: SVG path objects and parser
- [bob.learn.mlp]: Bob's Multi-layer Perceptron (MLP).
- [editdistance]: Fast implementation of the edit distance(Levenshtein distance).
- [cdtw]: dynamic time wrapping

See [here](http://www.lfd.uci.edu/~gohlke/pythonlibs/) for Windows binaries.

*Info: Since the multilayer_perceptron classes from scikit-learn are not yet included in the latest release (0.17.1), I have copied them. As soon as they are released, we can delete the code in folder mlp.*

## Classifier Results
### MLP                                                  
![MLP Training Neurons][fig1.1]
![MLP Training Algorithms][fig1.2]
###SVM
Scores for different kernels and confusion matrices

kernel | training score | training cross validation | test score |test cross validation
:----: | :------------: | :-----------------------: | :--------: |:-------------------:
linear | 1              | 0.910                     | 0.908      |0.913                
poly1  | 1              | 0.910                     | 0.908      |0.958                
poly4  | 1              | 0.955                     | 0.966      |0.946                

![linear kernel confusion matrix][fig2.1]
![ploy 3 kernel confusion matrix][fig2.2]
![poly 4 kernel confusion matrix][fig2.3]


## Key Word Search
The main project of the course is about implementing a solution for key word search in historical documents. 
### Pre-processing
Before extracting features, each word is pre-processed:
- remove clutter (small objects)
- find a word mask
- normalize the pixel intensities
- position all the words in a frame with uniform height (centering and scaling)

During this procedure the main assumption is that that the central part of the handwriting (i.e. small letters like a, e, 
i, ...) will be the predominant peak on the vertical projection of the pixels.

### Feature computation
Sliding window approach. Local descriptor includes:
- black-white transitions
- foreground fractions
- relative positions (top, bottom, centroid, center of mass)
- gray-scale moments

### Performance

dataset     | overall accuracy | accuracy with training samples | cpu time 
:---------: | :--------------: | :----------------------------: | :-------: 
training    | 0.47             | 0.47                           | 10h 30min      
validation  | 0.32             | 0.51                           | 8h 50min      
everything  | 0.42             | 0.48                           | 19h 20 min      

![accuracy vs. training samples][fig3.1]


[cython]: http://cython.org/
[ConfigParser]: https://docs.python.org/3/library/configparser.html
[Google Python Style Guide]: https://google.github.io/styleguide/pyguide.html
[numpy]: http://www.numpy.org/
[scipy]: http://www.scipy.org/
[scikit-image]: http://scikit-image.org/
[sklearn]: http://scikit-learn.org/
[sklearn.svm]: http://scikit-learn.org/stable/modules/svm.html
[sklearn.neural_network.multilayer_perceptron]: https://github.com/scikit-learn/scikit-learn/pull/3204
[svg.path]: https://pypi.python.org/pypi/svg.path
[bob.learn.mlp]: https://pypi.python.org/pypi/bob.learn.mlp
[editdistance]: https://github.com/aflc/editdistance
[cdtw]: https://github.com/honeyext/cdtw


[fig1.1]: https://raw.githubusercontent.com/dwettstein/pattern-recognition-2016/master/figs/mlp_neurons_630.png
[fig1.2]: https://raw.githubusercontent.com/dwettstein/pattern-recognition-2016/master/figs/mlp_main_algorithms.png

[fig2.1]: https://raw.githubusercontent.com/dwettstein/pattern-recognition-2016/master/figs/SVM_confusion-matrix_linear.png
[fig2.2]: https://raw.githubusercontent.com/dwettstein/pattern-recognition-2016/master/figs/SVM_confusion-matrix_poly_3.png
[fig2.3]: https://raw.githubusercontent.com/dwettstein/pattern-recognition-2016/master/figs/SVM_confusion-matrix_poly_4.png

[fig3.1]: https://raw.githubusercontent.com/dwettstein/pattern-recognition-2016/master/figs/kws_samples-vs-accuracy.png