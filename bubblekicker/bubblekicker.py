"""
S. Van Hoey
2016-06-06
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

        self.bubble_properties = None

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
        """change the channel"""
        self._channel_control(channel)
        self._channel = channel
        self.raw_image = self.raw_file[:, :, CHANNEL_CODE[self._channel]]
        self.current_image = self.raw_image.copy()
        self.logs.clear_log()
        print("Currently using channel {}".format(self._channel))

    def what_channel(self):
        """check the current working channel"""
        print(self._channel)

    @staticmethod
    def _channel_control(channel):
        """check if channel is either red, green, blue"""
        if channel not in ['red', 'green', 'blue']:
            raise NotAllowedChannel('Not a valid channel for '
                                    'RGB color scheme!')

    def edge_detect_canny_opencv(self, threshold=[0.01, 0.5]):
        """perform the edge detection algorithm of Canny on the image using
        the openCV package"""

        image = cv.Canny(self.current_image,
                         threshold[0],
                         threshold[1])

        self.current_image = image
        self.logs.add_log('edge-detect with thresholds {} -> {} '
                          '- opencv'.format(threshold[0], threshold[1]))
        return image

    def edge_detect_canny_skimage(self, sigma=3, threshold=[0.01, 0.5]):
        """perform the edge detection algorithm of Canny on the image"""
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
        """perform the edge detection algorithm of Canny on the image using
        the openCV package"""

        image = cv.adaptiveThreshold(self.current_image, 1,
                                     cv.ADAPTIVE_THRESH_MEAN_C,
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
        """clear the borders of the image"""

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
        """erode the image"""

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

    def calculate_bubble_properties(self):
        """provide a label for each bubble in the image"""

        nbubbles, marker_image = cv.connectedComponents(1 - self.current_image)
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
        self.bubble_properties = bubble_properties.copy()

        return nbubbles, marker_image, bubble_properties

    def filter_bubble_properties(self, rules=DEFAULT_FILTERS):
        """exclude bubbles based on a set of rules

        :return:
        """
        bubble_props = self.bubble_properties.copy()
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
        return bubble_props

    def show_distribution(self, which_property="equivalent_diameter",
                          bins=20):
        """calculate and create the distribution plot"""
        # using the elf.current_image => calculate and derive distribution plot
        # you could opt to have the plot function itself outside the class
        # as this makes it more general
        fig, ax = plt.subplots()
        n, bins, patches = ax.hist(self.bubble_properties[which_property],
                                   bins, normed=1, cumulative=True)

        return fig, ax

