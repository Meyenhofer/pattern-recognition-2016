import math
import matplotlib.pyplot as plt


def show_digit(v):
    """
    show the digit assuming the vector can be reshaped in a square image
    :param v: pixel values n x 1
    """
    d = math.sqrt(v.shape[0])
    r = v.reshape([d, d])
    return plt.imshow(r)
