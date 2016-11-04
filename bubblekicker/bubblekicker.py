"""
S. Van Hoey
2016-06-06
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from skimage.data import imread
from skimage.feature import canny
from skimage.segmentation import clear_border
from skimage.morphology import dilation, rectangle
from skimage.measure import regionprops

import cv2 as cv

from bubblekicker.utils import (calculate_convexity,
                                calculate_circularity_reciprocal)

CHANNEL_CODE = {'red': 0, 'green': 1, 'blue': 2}
DEFAULT_FILTERS = {'circularity_reciprocal': {'min': 0.2, 'max': 1.6},
                   'convexity': {'min': 0.92}}


class NotAllowedChannel(Exception):
    """
    Exception placeholder for easier debugging.
    """
    pass


class Logger(object):
    """
    Log the sequence of log statements performed
    """
    def __init__(self):
        self.log = []

    def add_log(self, message):
        """add a log statement to the sequence"""
        self.log.append(message)

    def get_last_log(self):
        return self.log[-1]

    def print_log_sequence(self):
        print("Steps undertaken since from raw image:")
        print("\n".join(self.log))
        print("\n")

    def clear_log(self):
        """clear all the logs"""
        self.log = []


def batchbubblekicker(data_path, channel, pipeline, *args):
    """
    Given a folder with processable files and a channel to use, a sequence
    of steps class as implemented in the pipelines.py file will be applied on
    each of the individual images

    :param data_path: folder containing images to process
    :param channel: green | red | blue
    :param pipeline: class from pipelines.py to use as processing sequence
    :param args: arguments required by the pipeline
    :return: dictionary with for each file the output binary image
    """
    results = {}

    for imgfile in os.listdir(data_path):
        current_bubbler = pipeline(os.path.join(data_path, imgfile),
                                   channel=channel)
        results[imgfile] = current_bubbler.run(*args)
    return results


class BubbleKicker(object):

    def __init__(self, filename, channel='red'):
        """
        This class contains a set of functions that can be applied to a
        bubble image in order to derive a binary bubble-image and calculate the
        statistics/distribution

        :param filename: image file name
        :param channel: green | red | blue
        """

        self.raw_file = self._read_image(filename)
        self.logs = Logger()

        self._channel_control(channel)
        self._channel = channel

        self.raw_image = self.raw_file[:, :, CHANNEL_CODE[self._channel]]
        self.current_image = self.raw_image.copy()

    @staticmethod
    def _read_image(filename):
        """read the image from a file and store
        an RGB-image MxNx3
        """
        image = imread(filename)
        return image

    def reset_to_raw(self):
        """make the current image again the raw image"""
        self.current_image = self.raw_image.copy()
        self.logs.clear_log()

    def switch_channel(self, channel):
        """change the color channel"""
        self._channel_control(channel)
        self._channel = channel
        self.raw_image = self.raw_file[:, :, CHANNEL_CODE[self._channel]]
        self.current_image = self.raw_image.copy()
        self.logs.clear_log()
        print("Currently using channel {}".format(self._channel))

    def what_channel(self):
        """check the current working channel (R, G or B?)"""
        print(self._channel)

    @staticmethod
    def _channel_control(channel):
        """check if channel is either red, green, blue"""
        if channel not in ['red', 'green', 'blue']:
            raise NotAllowedChannel('Not a valid channel for '
                                    'RGB color scheme!')

    def edge_detect_canny_opencv(self, threshold=[0.01, 0.5]):
        """perform the edge detection algorithm of Canny on the image using
        the openCV package. Thresholds are respectively min and max threshodls for building 
	the gaussian."""

        image = cv.Canny(self.current_image,
                         threshold[0],
                         threshold[1])

        self.current_image = image
        self.logs.add_log('edge-detect with thresholds {} -> {} '
                          '- opencv'.format(threshold[0], threshold[1]))
        return image

    def edge_detect_canny_skimage(self, sigma=3, threshold=[0.01, 0.5]):
        """perform the edge detection algorithm of Canny on the image using scikit package"""
        image = canny(self.current_image,
                      sigma=sigma,
                      low_threshold=threshold[0],
                      high_threshold=threshold[1])

        self.current_image = image

        # append function to logs
        self.logs.add_log('edge-detect with '
                          'thresholds {} -> {} and sigma {} '
                          '- skimage'.format(threshold[0],
                                             threshold[1],
                                             sigma))
        return image

    def adaptive_threshold_opencv(self, blocksize=91, cvalue=18):
        """perform the edge detection algorithm of Canny on the image using an adaptive 
	threshold method for which the user can specify width of the window of action 
	and a C value used as reference for building the gaussian distribution. This function 
	uses the openCV package
	
	Parameters
        ----------
	blocksize:

	cvalue:


	"""

        image = cv.adaptiveThreshold(self.current_image, 1,
                                     cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv.THRESH_BINARY, blocksize, cvalue)

        self.current_image = image
        self.logs.add_log('adaptive threshold bubble detection '
                          'with blocksize {} and cvalue {} '
                          '- opencv'.format(blocksize, cvalue))
        return image

    def dilate_opencv(self, footprintsize=3):
        """perform the dilation of the image"""

        # set up structuring element with footprintsize
        kernel = np.ones((footprintsize, footprintsize), np.uint8)

        # perform algorithm with given environment,
        # store in same memory location
        image = cv.dilate(self.current_image, kernel, iterations=1)

        # update current image
        self.current_image = image

        # append function to logs
        self.logs.add_log('dilate with footprintsize {} '
                          '- opencv'.format(footprintsize))
        return image

    def dilate_skimage(self):
        """perform the dilation of the image"""

        # set up structuring element
        # (@Giacomo, is (1, 90) and (1, 0) different? using rectangle here...
        struct_env = rectangle(1, 1)

        # perform algorithm with given environment,
        # store in same memory location
        image = dilation(self.current_image, selem=struct_env,
                         out=self.current_image)

        # update current image
        self.current_image = image

        # append function to logs
        self.logs.add_log('dilate - skimage')

        return image

    def fill_holes_opencv(self):
        """fill the holes of the image"""
        # perform algorithm
        h, w = self.current_image.shape[:2]  # stores image sizes
        mask = np.zeros((h + 2, w + 2), np.uint8)
        # floodfill operates on the saved image itself
        cv.floodFill(self.current_image, mask, (0, 0), 0)

        # append function to logs
        self.logs.add_log('fill holes - opencv')
        return self.current_image

    def clear_border_skimage(self, buffer_size=3, bgval=1):
        """clear the borders of the image using a belt of pixels definable in buffer_size and 
	asign a pixel value of bgval
	
	Parameters
        ----------
        buffer_size: int
	indicates the belt of pixels around the image border that should be considered to 
	eliminate touching objects (default is 3)
	
	bgvalue: int
	all touching objects are set to this value (default is 1)
	"""

        # perform algorithm
        image_inv = cv.bitwise_not(self.current_image)
        image = clear_border(image_inv, buffer_size=buffer_size, bgval=bgval)

        # update current image
        self.current_image = image

        # append function to logs
        self.logs.add_log('clear border with buffer size {} and bgval {} '
                          '-  skimage'.format(buffer_size, bgval))
        return image

    def erode_opencv(self, footprintsize=1):
        """erode detected edges with a given footprint. This function is meant to be used after dilation of the edges so to reset the original edge."""

        kernel = np.ones((footprintsize, footprintsize), np.uint8)
        image = cv.erode(self.current_image, kernel, iterations=1)

        # update current image
        self.current_image = image

        # append function to logs
        self.logs.add_log('erode with footprintsize {} '
                          '- opencv'.format(footprintsize))
        return image

    def what_have_i_done(self):
        """ print the current log statements as a sequence of
        performed steps"""
        self.logs.print_log_sequence()

    def plot(self):
        """plot the current image"""
        fig, ax = plt.subplots()
        ax.imshow(self.current_image, cmap=plt.cm.gray)
        if len(self.logs.log) > 0:
            ax.set_title(self.logs.log[-1])
        return fig, ax


def _bubble_properties_table(binary_image):
    """provide a label for each bubble in the image"""

    nbubbles, marker_image = cv.connectedComponents(1 - binary_image)
    props = regionprops(marker_image)
    bubble_properties = \
        pd.DataFrame([{"label": bubble.label,
                       "area": bubble.area,
                       "centroid": bubble.centroid,
                       "convex_area": bubble.convex_area,
                       "equivalent_diameter": bubble.equivalent_diameter,
                       "perimeter": bubble.perimeter} for bubble in props])

    bubble_properties["convexity"] = \
        calculate_convexity(bubble_properties["perimeter"],
                            bubble_properties["area"])
    bubble_properties["circularity_reciprocal"] = \
        calculate_circularity_reciprocal(bubble_properties["perimeter"],
                                         bubble_properties["area"])

    bubble_properties = bubble_properties.set_index("label")

    return nbubbles, marker_image, bubble_properties


def _bubble_properties_filter(property_table, id_image,
                              rules=DEFAULT_FILTERS):
    """exclude bubbles based on a set of rules

    :return:
    """
    bubble_props = property_table.copy()
    all_ids = bubble_props.index.tolist()

    for prop_name, ruleset in rules.items():
        print(ruleset)
        for rule, value in ruleset.items():
            if rule == 'min':
                bubble_props = \
                    bubble_props[bubble_props[prop_name] > value]
            elif rule == 'max':
                bubble_props = \
                    bubble_props[bubble_props[prop_name] < value]
            else:
                raise Exception("Rule not supported, "
                                "use min or max as filter")

    removed_ids = [el for el in all_ids if el
                   not in bubble_props.index.tolist()]
    for idb in removed_ids:
        id_image[id_image == idb] = 0

    return id_image, bubble_props


def bubble_properties_calculate(binary_image,
                                rules=DEFAULT_FILTERS):
    """

    :param binary_image:
    :param rules:
    :return:
    """
    # get the bubble identifications and properties
    nbubbles, id_image, \
        prop_table = _bubble_properties_table(binary_image)
    # filter based on the defined rules
    id_image, properties = _bubble_properties_filter(prop_table,
                                                     id_image, rules)
    return id_image, properties


def bubble_properties_plot(property_table,
                           which_property="equivalent_diameter",
                           bins=20):
    """calculate and create the distribution plot"""
    fontsize_labels = 14.
    formatter = FuncFormatter(
        lambda y, pos: "{:d}%".format(int(round(y * 100))))
    fig, ax1 = plt.subplots()
    ax1.hist(property_table[which_property], bins,
             normed=0, cumulative=False, histtype='bar',
             color='gray', ec='white')
    ax1.get_xaxis().tick_bottom()

    # left axis - histogram
    ax1.set_ylabel(r'Frequency', color='gray',
                   fontsize=fontsize_labels)
    ax1.spines['top'].set_visible(False)

    # right axis - cumul distribution
    ax2 = ax1.twinx()
    ax2.hist(property_table[which_property],
             bins, normed=1, cumulative=True,
             histtype='step', color='k', linewidth= 3.)
    ax2.yaxis.set_major_formatter(formatter)
    ax2.set_ylabel(r'Cumulative percentage (%)', color='k',
                   fontsize=fontsize_labels)
    ax2.spines['top'].set_visible(False)
    ax2.set_ylim(0, 1.)

    # additional options
    ax1.set_xlim(0, property_table[which_property].max())
    ax1.tick_params(axis='x', which='both', pad=10)
    ax1.set_xlabel(which_property)

    return fig, (ax1, ax2)


