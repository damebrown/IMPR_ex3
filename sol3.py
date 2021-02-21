import os
import numpy as np
from scipy.ndimage.filters import convolve
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from imageio import imread
from skimage import img_as_float64
from skimage.color import rgb2gray


# ================
# Global variables
# ================

# number of dimensions in a RGB image
RGB_LEN = 3
# representation code to return a laplacian pyramid
IS_LAPLACIAN = 0
# representation code to return a gaussian pyramid
IS_GAUSSIAN = 1
# representation code for a gray scale image
GRAY_REP = 1
# the minimal size of a pyramid level
MIN_IMAGE_DIM = 16
# basic gaussian kernel
GAUSSIAN_KERNEL = np.matrix([1, 1]).astype("float64")

# ==========================
# Pyramid Building Functions
# ==========================


def read_image(filename, representation):
    """
    Reads an image file and converts it into a given representation
    :param filename: The filename of an image on disk (could be grayscale or RGB).
    :param representation: Representation code, either 1 or 2 defining whether the output should be a grayscale
            image (1) or an RGB image (2).
    :return: an image represented by a matrix of type np.float64 with intensities normalized to [0,1]
    """
    rgb_img = imread(filename)
    rgb_img = img_as_float64(rgb_img)
    if representation == GRAY_REP:
        rgb_img = rgb2gray(rgb_img)
    return rgb_img


def blur(im, filter_vec):
    """
    blurs the input image with the input filter vector
    :param im: the image to blur
    :param filter_vec: the filter to blur with
    :return: the blurred image
    """
    return convolve(convolve(im, filter_vec, mode = 'mirror'), filter_vec.T, mode = 'mirror')


def gaussian_factory(filter_size):
    """
    an aid function for the gaussian pyramid functions. generates a gaussian kernel of the wanted size.
    :param filter_size: the wanted size
    :return: a (filter_size, filter_size) shape gaussian kernel
    """
    if filter_size == 1 or filter_size % 2 == 0:
        return np.matrix([1])
    base_vector = GAUSSIAN_KERNEL
    while len(base_vector[0]) < filter_size:
        base_vector = convolve2d(base_vector, GAUSSIAN_KERNEL)
    return base_vector / sum(base_vector[0])


def laplacian_factory(im, filter_vec):
    """
    An aid function for the laplacian pyramid functions. generates a laplacian kernel of the wanted size.
    :param filter_vec: the wanted size
    :param im: the image to make into a laplacian level
    :return: a (filter_size, filter_size) shape laplacian kernel
    """
    reduced = reduce(blur(im, filter_vec))
    expanded = expand(reduced, filter_vec)
    return im - expanded


def reduce_row(row):
    """
    This function does the reducing of a single row in an image.
    :param row:
    :return:
    """
    new_row = np.zeros((int(row.shape[0] / 2), ))
    for pixel in range(len(new_row)):
        new_row[pixel] = row[2 * pixel]
    return new_row


def reduce(im):
    """
    Reduces a the inputed image.
    :param im: the image to reduce
    :return: the reduced image
    """
    shape = (int(im.shape[0] / 2), int(im.shape[1] / 2))
    new_level = np.zeros(shape)
    for row in range(new_level.shape[0]):
        new_level[row] = reduce_row(im[2 * row])
    return new_level


def input_validity_check(im, max_levels, filter_size):
    """
    Checks the validity of the arguments to the build_pyramid functions
    :param im: the image to make a pyramid from
    :param max_levels: the maximum number of levels in the pyramid
    :param filter_size: the filter size
    :return: true iff the input is valid, false otherwise
    """
    return len(im.shape) == 2 and max_levels >= 1 and max_levels % 1 == 0 and filter_size % 2 != 0


def build_pyramid_general(im, max_levels, filter_size, is_gaussian):
    """
    a general pyramid building function that receives data about whether the function was called from the laplacian
    pyramid function or the gaussian, and returns accordingly.
    :param im: the image to make a pyramid from
    :param max_levels: the maximum number of levels in the pyramid
    :param filter_size: the filter size
    :param is_gaussian: 1 if the function was called from the build_pyramid_gaussian, 0 otherwise
    :return: the pyramid respective to the is_gaussian argument
    """
    if not input_validity_check(im, max_levels, filter_size):
        raise ValueError("Error: invalid values received as arguments to build_gaussian_pyramid function.")
    level, filter_vec = im, gaussian_factory(filter_size)
    if is_gaussian == IS_GAUSSIAN:
        pyr = [im]
    else:
        pyr = [laplacian_factory(level, filter_vec)]
    while len(pyr) < max_levels and min(level.shape[0], level.shape[1]) >= MIN_IMAGE_DIM:
        level = reduce(blur(level, filter_vec))
        if is_gaussian == IS_LAPLACIAN:
            pyr.append(laplacian_factory(level, filter_vec))
        elif is_gaussian == IS_GAUSSIAN:
            pyr.append(level)
    return pyr, filter_vec


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Constructs a Gaussian pyramid of a given image
    :param im: a grayscale image with double values in [0, 1] (e.g. the output of ex1’s read_image with the
                representation set to 1).
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
            in constructing the pyramid filter.
    :return: pyr, filter_vec where pyr is a standard python array with max length of max_levels, where each element is
            a grayscale image in descending resolution, and filter_vec is a normalized (to range [0,1]) row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    return build_pyramid_general(im, max_levels, filter_size, IS_GAUSSIAN)


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Constructs a Laplacian pyramid of a given image.
    :param im: a grayscale image with double values in [0, 1] (e.g. the output of ex1’s read_image with the
                representation set to 1).
    :param max_levels: the maximal number of levels1 in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
            in constructing the pyramid filter.
    :return: pyr, filter_vec where pyr is a standard python array with max length of max_levels, where each element is
            a grayscale image in descending resolution, and filter_vec is a normalized (to range [0,1]) row vector of
            shape (1, filter_size) used for the pyramid construction.
    """
    return build_pyramid_general(im, max_levels, filter_size, IS_LAPLACIAN)


# =====================================
# Building Image from Pyramid Functions
# =====================================


def expand(im, filter_vec):
    """
    expands the image recieved.
    :return: the expanded image
    """
    shape = im.shape[0] * 2, im.shape[1] * 2
    expanded = np.zeros(shape)
    expanded[::2, ::2] = im
    return blur(expanded, 2 * filter_vec)


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    the Laplacian pyramid that are generated by
    :param lpyr: the Laplacian pyramid that is generated by builid_laplacian_pyramid function
    :param filter_vec: the filter_vec that is generated by builid_laplacian_pyramid function
    :param coeff: a python list. The list length is the same as the number of levels in the pyramid lpyr. coefficants
                to multiply lpyr's elements by.
    :return: an image from its Laplacian Pyramid
    """
    normal_coeff = [lpyr[i] * coeff[i] for i in range(len(lpyr))]
    g, expanded = normal_coeff[-1], 0
    for i in range(len(lpyr) - 1, 0, -1):
        expanded = expand(g, filter_vec)
        g = expanded + normal_coeff[i - 1]
    return g


# ============================
# Displaying Pyramid Functions
# ============================


def calculate_shape(sub_pyr):
    """
    calculates the shape of the image of pyramid levels
    :param sub_pyr: an array of pyramid levels
    :return: the shape of the wanted image
    """
    height = sub_pyr[0].shape[0]
    width = 0
    for level in sub_pyr:
        width += level.shape[1]
    return height, width


def stretch_levels(pyr):
    """
    stretches the levels of pyr to [0,1]
    :param pyr: the pyramid array to stretch.
    :return: the stretched pyramid/
    """
    for level in range(len(pyr)):
        min_pixel, max_pixel = np.amin(pyr[level]), np.amax(pyr[level])
        if min_pixel < 0:
            diff = max_pixel - min_pixel
            if diff:
                pyr[level] = (-pyr[level] + min_pixel) / -diff
        else:
            diff = max_pixel - min_pixel
            if diff:
                pyr[level] = (pyr[level] - min_pixel) / diff
    return pyr


def merge_levels(sub_pyr, black):
    """
    builds the image to display in the render_pyramid function.
    :param sub_pyr: the pyramid to display
    :param black: the black image using as background
    :return: the image
    """
    width = 0
    for level in sub_pyr:
        black[0:level.shape[0], width:width + level.shape[1]] = level
        width += level.shape[1]
    return black


def render_pyramid(pyr, levels):
    """
    returns a black image in the corresponding size
    :param pyr: either a Gaussian or Laplacian pyramid.
    :param levels: the number of levels to present in the result
    :return: a single black image in which the pyramid levels of the given pyramid pyr are stacked horizontally
    """
    stretched_pyr = stretch_levels(pyr)
    shape = calculate_shape(stretched_pyr[:levels])
    return merge_levels(stretched_pyr[:levels], np.zeros(shape))


def display_pyramid(pyr, levels):
    """
    Uses render_pyramid to internally render and then display the stacked pyramid image using plt.imshow().
    :param pyr: either a Gaussian or Laplacian pyramid.
    :param levels: the number of levels to present in the result
    """
    plt.figure()
    plt.imshow(render_pyramid(pyr, levels), cmap = 'gray')
    plt.show()


# ==========================
# Pyramid Blending Examples Functions
# ==========================


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    Implementation of pyramid blending as described in the lecture.
    :param im1, im2: two input grayscale images to be blended. Both have the same dimension.
    :param mask: a boolean (of dtype np.bool) mask containing True and False representing which parts
                of im1 and im2 should appear in the resulting im_blend, while True is 1, False is 0. Has the same
                dimension as im1 and im2.
    :param max_levels: the max_levels parameter you should use when generating the Gaussian and Laplacian
                pyramids.
    :param filter_size_im: the size of the Gaussian filter which defining the filter used in the construction of the
                Laplacian pyramids of im1 and im2.
    :param filter_size_mask: the size of the Gaussian filter which defining the filter used in the construction of the
                Gaussian pyramid of mask.
    :return: im_blend: valid grayscale image in the range [0, 1].
    """
    lapyr1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)[0]
    lapyr2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]
    mask_pyr, filter_vec = build_gaussian_pyramid(img_as_float64(mask), max_levels, filter_size_mask)
    mask_pyr = stretch_levels(mask_pyr)
    lap_out = []
    for i in range(len(mask_pyr)):
        level = np.multiply(mask_pyr[i], lapyr1[i]) + np.multiply((1 - mask_pyr[i]), (lapyr2[i]))
        lap_out.append(level)
    return laplacian_to_image(lap_out, filter_vec, [1 for _ in range(len(lap_out))])


def show_images1(im_arr):
    """
    shows the images of the first example
    :param im_arr: array of images to show
    """
    fig = plt.figure(figsize = (2, 2))
    columns = 2
    rows = 2
    for i in range(4):
        fig.add_subplot(rows, columns, i + 1)
        if i == 2:
            plt.imshow(im_arr[i], cmap = 'gray')
        else:
            plt.imshow(im_arr[i])
    plt.show()


def show_images2(im_arr):
    """
    shows the images of the second example
    :param im_arr: array of images to show
    """
    fig = plt.figure(figsize = (2, 2))
    columns = 2
    rows = 2
    for i in range(4):
        fig.add_subplot(rows, columns, i + 1)
        if i == 2 or i == 1:
            plt.imshow(im_arr[i], cmap = 'gray')
        else:
            plt.imshow(im_arr[i])
    plt.show()


def blending_example1():
    """
    an example of pyramid blending
    :return: im1, im2, mask, im_blend- the two images blended, the mask and the blended image
    """
    im1, im2 = read_image(relpath('blend.jpg'), 2), read_image(relpath('pyr_cut.jpg'), 2)
    mask = read_image(relpath('mask1.jpg'), 1)
    mask[mask < 1] = 0
    im_blend = np.empty_like(im1)
    max_levels = 7
    for color in range(RGB_LEN):
        im_blend[:, :, color] = pyramid_blending(im1[:, :, color], im2[:, :, color], mask, max_levels, 19, 3)
        im_blend[:, :, color] = stretch_levels([im_blend[:, :, color]])[0]
    show_images1([im1, im2, mask, im_blend])
    return im1, im2, mask.astype(bool), im_blend


def blending_example2():
    """
    an example of pyramid blending
    :return: im1, im2, mask, im_blend- the two images blended, the mask and the blended image
    """
    im1, im2 = read_image(relpath('trump.jpg'), 2), read_image(relpath('sigma.jpg'), 1)
    mask = read_image(relpath('mask2.jpg'), 1)
    mask[mask < 1] = 0
    im_blend = np.empty_like(im1)
    max_levels = 5
    for color in range(RGB_LEN):
        im_blend[:, :, color] = pyramid_blending(im1[:, :, color], im2, mask, max_levels, 9, 9)
        im_blend[:, :, color] = stretch_levels([im_blend[:, :, color]])[0]
    show_images2([im1, im2, mask, im_blend])
    return im1, im2, mask.astype(bool), im_blend


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


blending_example1()
blending_example2()
