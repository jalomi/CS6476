import math
import numpy as np
import cv2
import sys

# # Implement the functions below.


def extract_red(image):
    """ Returns the red channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the red channel.
    """
    image_tmp = np.copy(image)
    return image_tmp[:, :, 2]


def extract_green(image):
    """ Returns the green channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the green channel.
    """

    image_tmp = np.copy(image)
    return image_tmp[:, :, 1]


def extract_blue(image):
    """ Returns the blue channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the blue channel.
    """

    image_tmp = np.copy(image)
    return image_tmp[:, :, 0]


def swap_green_blue(image):
    """ Returns an image with the green and blue channels of the input image swapped. It is highly
    recommended to make a copy of the input image in order to avoid modifying the original array.
    You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 3D array with the green and blue channels swapped.
    """

    image_tmp = np.copy(image)
    result = np.copy(image)
    blue = extract_blue(image_tmp)
    green = extract_green(image_tmp)


    image_tmp[:, :, 0] = green
    image_tmp[:, :, 1] = blue

    return image_tmp


def copy_paste_middle(src, dst, shape):
    """ Copies the middle region of size shape from src to the middle of dst. It is
    highly recommended to make a copy of the input image in order to avoid modifying the
    original array. You can do this by calling:
    temp_image = np.copy(image)

        Note: Assumes that src and dst are monochrome images, i.e. 2d arrays.

        Note: Where 'middle' is ambiguous because of any difference in the oddness
        or evenness of the size of the copied region and the image size, the function
        rounds downwards.  E.g. in copying a shape = (1,1) from a src image of size (2,2)
        into an dst image of size (3,3), the function copies the range [0:1,0:1] of
        the src into the range [1:2,1:2] of the dst.

    Args:
        src (numpy.array): 2D array where the rectangular shape will be copied from.
        dst (numpy.array): 2D array where the rectangular shape will be copied to.
        shape (tuple): Tuple containing the height (int) and width (int) of the section to be
                       copied.

    Returns:
        numpy.array: Output monochrome image (2D array)
    """

    src_tmp = np.copy(src)
    dst_tmp = np.copy(dst)

    src_size = src_tmp.shape
    dst_size = dst_tmp.shape

    src_x_mid = int(src_size[0] / 2)
    src_y_mid = int(src_size[1] / 2)
    src_x_start = src_x_mid - int(shape[0] / 2)
    src_y_start = src_y_mid - int(shape[1] / 2)
    src_x_end = src_x_start + shape[0]
    src_y_end = src_y_start + shape[1]

    src_patch = src_tmp[src_x_start:src_x_end, src_y_start:src_y_end]

    dst_x_mid = int(dst_size[0] / 2)
    dst_y_mid = int(dst_size[1] / 2)
    dst_x_start = dst_x_mid - int(shape[0] / 2)
    dst_y_start = dst_y_mid - int(shape[1] / 2)
    dst_x_end = dst_x_start + shape[0]
    dst_y_end = dst_y_start + shape[1]

    dst_tmp[dst_x_start:dst_x_end, dst_y_start:dst_y_end] = src_patch

    return dst_tmp


def image_stats(image):
    """ Returns the tuple (min,max,mean,stddev) of statistics for the input monochrome image.
    In order to become more familiar with Numpy, you should look for pre-defined functions
    that do these operations i.e. numpy.min.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.

    Returns:
        tuple: Four-element tuple containing:
               min (float): Input array minimum value.
               max (float): Input array maximum value.
               mean (float): Input array mean / average value.
               stddev (float): Input array standard deviation.
    """

    image_tmp = np.copy(image)
    image_tmp = image_tmp.astype(float)

    img_min = np.min(image_tmp)
    img_max = np.max(image_tmp)
    img_mean = np.mean(image_tmp)
    img_stddev = np.std(image_tmp)

    return (img_min, img_max, img_mean, img_stddev)


def center_and_normalize(image, scale):
    """ Returns an image with the same mean as the original but with values scaled about the
    mean so as to have a standard deviation of "scale".

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        scale (int or float): scale factor.

    Returns:
        numpy.array: Output 2D image.
    """

    image_tmp = np.copy(image)

    #convert to float per prompt's suggestion
    image_tmp = image_tmp.astype(float)

    img_min, img_max, img_mean, img_stddev = image_stats(image_tmp)

    image_tmp -= img_mean
    image_tmp /= img_stddev
    image_tmp *= scale
    image_tmp += img_mean

    # convert back before returning
    return image_tmp.astype(int)



def shift_image_left(image, shift):
    """ Outputs the input monochrome image shifted shift pixels to the left.

    The returned image has the same shape as the original with
    the BORDER_REPLICATE rule to fill-in missing values.  See

    http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/copyMakeBorder/copyMakeBorder.html?highlight=copy

    for further explanation.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        shift (int): Displacement value representing the number of pixels to shift the input image.
            This parameter may be 0 representing zero displacement.

    Returns:
        numpy.array: Output shifted 2D image.
    """

    image_tmp = np.copy(image)
    out_img = np.copy(image)

    if (shift):
        patch = image_tmp[:,shift:].copy()
        out_img = cv2.copyMakeBorder(patch, 0, 0, 0, shift, cv2.BORDER_REPLICATE)

    return out_img




def difference_image(img1, img2):
    """ Returns the difference between the two input images (img1 - img2). The resulting array must be normalized
    and scaled to fit [0, 255].

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        img1 (numpy.array): Input 2D image.
        img2 (numpy.array): Input 2D image.

    Returns:
        numpy.array: Output 2D image containing the result of subtracting img2 from img1.
    """

    image1_tmp = np.copy(img1)
    image2_tmp = np.copy(img2)

    image1_tmp = image1_tmp.astype(float)
    image2_tmp = image2_tmp.astype(float)

    image_diff = (image1_tmp - image2_tmp)

    range = image_diff.max() - image_diff.min()

    # apparently Bonnie can test w/ the image being all the same value. (Don't divide by 0)
    if (range):
        result = (image_diff - image_diff.min()) / (range) * 255

    return result


def add_noise(image, channel, sigma):
    """ Returns a copy of the input color image with Gaussian noise added to
    channel (0-2). The Gaussian noise mean must be zero. The parameter sigma
    controls the standard deviation of the noise.

    The returned array values must not be clipped or normalized and scaled. This means that
    there could be values that are not in [0, 255].

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): input RGB (BGR in OpenCV) image.
        channel (int): Channel index value.
        sigma (float): Gaussian noise standard deviation.

    Returns:
        numpy.array: Output 3D array containing the result of adding Gaussian noise to the
            specified channel.
    """

    image_tmp = np.copy(image)

    image_channel = image_tmp[:,:,channel]

    #convert to float
    image_channel = image_channel.astype(float)

    noise = np.random.normal(0, sigma, image_channel.shape)

    image_channel += noise

    image_tmp[:,:,channel] = image_channel.astype(int)

    return image_tmp
