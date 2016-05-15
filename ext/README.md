# Dynamic Time Warping C extension

A C implementation of the Dynamic Time Warping algorithm with a Python binding.
The *Euclidean distance* is used as the distance metric.

***Requires Python 3***.


## Installation

```sh
python setup.py install
```

## Example

```python
import numpy as np
from dtwextension import dtwdistance

arr1 = np.array([[1.0, 2.0], [2.4, 1.7], [9.9, 0.0]])
arr2 = np.array([[2.4, 2.7], [1.2, 3.3], [0.2, 5.5]])
arr3 = np.random.rand(10, 3) # randomly initialized 10 x 3 array

dtwdistance(arr1, arr2) # => 14.71603230999898
dtwdistance(arr1, arr3) # works with different input sizes
```
