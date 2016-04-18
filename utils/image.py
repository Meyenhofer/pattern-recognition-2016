import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import grid_points_in_poly
from skimage.transform import AffineTransform, warp

def crop(image, polygon):
    new_image = np.copy(image)
    xmin, xmax, ymin, ymax = polygon_border(polygon)
    included = grid_points_in_poly(new_image.shape, polygon)
    # set values outside the polygon to white
    new_image[~included] = 255
    cropped_image = new_image[xmin:xmax, ymin:ymax]

    return cropped_image

def polygon_border(polygon):
    xmin = polygon[0][0]
    xmax = 0
    ymin = polygon[0][1]
    ymax = 0
    for p in polygon:
        if xmin > p[0]:
            xmin = p[0]
        if xmax < p[0]:
            xmax = p[0]
        if ymin > p[1]:
            ymin = p[1]
        if ymax < p[1]:
            ymax = p[1]

    return [round(v) for v in [xmin, xmax, ymin, ymax]]

def binarize(image):
    threshold = threshold_otsu(image)

    return image > threshold

def skew_correction(image, shear=0.2):
    transform = AffineTransform(shear=shear)

    return warp(image, transform)
