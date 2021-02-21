import numpy as np
from scipy.ndimage.filters import convolve
import sol3
from imageio import imread
from skimage import img_as_float64
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

# representation code for a gray scale image
GRAY_REP = 1
# constant to normalize a [0,255] image
NORMALIZE_CONST = 255


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
    return rgb_img / NORMALIZE_CONST


def show_pyr(pyr):
    # for i in range(len(pyr)):
    #     plt.figure()
    #     plt.imshow(pyr[i], cmap = 'gray')
    #     plt.show()

    fig, a = plt.subplots(1, len(pyr))
    for j in range(len(pyr)):
        a[j].imshow(pyr[j], cmap = 'gray')
    plt.show()


def test_gauss():
    im = read_image('s.png', GRAY_REP)
    pyr, filter = sol3.build_gaussian_pyramid(im, 6, 5)
    sol3.display_pyramid(pyr, 6)


def test_laplac():
    im = read_image('s.png', GRAY_REP)
    return sol3.build_laplacian_pyramid(im, 5, 5)


def test_blend(im1, im2, mask, max_levels, mask_size, filter_size):
    a, b = read_image(im1, 1), read_image(im2, 1)
    mask = read_image(mask, 1)
    # mask = sol3.stretch_levels([mask])[0]
    # sol3.dis(mask)
    # lapyr1 = sol3.build_laplacian_pyramid(a, max_levels, 5)[0]
    # lapyr2 = sol3.build_laplacian_pyramid(b, max_levels, 5)[0]
    # mask_pyr, filter_vec = sol3.build_gaussian_pyramid(img_as_float64(mask), max_levels, 5)
    # lap_out = []
    # i = 1
    # level = np.multiply(mask_pyr[i], lapyr1[i]) + np.multiply((1 - mask_pyr[i]), (lapyr2[i]))
    # a = np.multiply(mask_pyr[i], lapyr1[i])
    # sol3.dis(np.multiply(mask_pyr[i], lapyr2[i]))
    # b = lapyr2[i] - np.multiply(mask_pyr[i], lapyr2[i])
    # lap_out.append(level)
    # sol3.dis(b)
    # sol3.dis(level)
    # lap_out.append(np.multiply(mask_pyr[i], lapyr1[i]) + np.multiply((1 - mask_pyr[i]), (lapyr2[i])))
    # display_pyramid(lap_out, max_levels)
    # return np.clip(sol3.laplacian_to_image(lap_out, filter_vec, [1 for _ in range(len(lap_out))]), 0, 1)

    # a = sol3.expand(sol3.expand(a, sol3.gaussian_factory(filter_size)), sol3.gaussian_factory(filter_size))
    # b = sol3.expand(sol3.expand(b, sol3.gaussian_factory(filter_size)), sol3.gaussian_factory(filter_size))
    # mask = sol3.expand(sol3.expand(mask, sol3.gaussian_factory(filter_size)), sol3.gaussian_factory(filter_size))
    plt.figure()
    im_blend = sol3.pyramid_blending(a, b, mask, max_levels,
                                     filter_size, mask_size)
    plt.imshow(im_blend, cmap = 'gray')
    plt.show()
    return


def dis(im, text):
    plt.figure()
    plt.suptitle(text)
    plt.imshow(im)
    plt.show()


im1, im2 = read_image('blend.jpg', 2), read_image('pyr_cut.jpg', 2)
mask = read_image('mask1.jpg', 1)
for i in range(1, 12):
    im_blend = np.empty_like(im1)
    max_levels = 7
    for color in range(sol3.RGB_LEN):
        # im_blend[:, :, color] = sol3.pyramid_blending(im1[:, :, color], im2[:, :, color], mask, max_levels,  2 * i + 1,  2 * i + 1)
        im_blend[:, :, color] = sol3.pyramid_blending(im1[:, :, color], im2[:, :, color], mask, i,  5,  5)
        im_blend[:, :, color] = sol3.stretch_levels([im_blend[:, :, color]])[0]
    dis(im_blend, "filter size is" + str(i))



def old_test():
    # lapyr, lap_filter = test_laplac()
    # test_gauss()
    # sol3.display_pyramid(lapyr, 5)
    # im1 = sol3.laplacian_to_image(pyr, filter, [1 for _ in range(len(pyr))])
    # sol3.display_pyramid([read_image('s.png', GRAY_REP)] + [im1], len(im1))
    # # fig, a = plt.subplots(1, 2)
    # a[0].imshow(im1, cmap = 'gray')
    # plt.suptitle('with coeff')
    # a[1].imshow(read_image('s.png', GRAY_REP), cmap = 'gray')
    # a[2].imshow(im1, cmap = 'gray')
    # plt.suptitle('without coeff')
    # plt.show()
    pass
