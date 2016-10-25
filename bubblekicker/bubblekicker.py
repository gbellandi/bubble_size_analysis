"""
S. Van Hoey
2016-06-06
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage.data import imread
from skimage.feature import canny
from skimage.segmentation import clear_border
from skimage.morphology import dilation, rectangle

import cv2 as cv

CHANNEL_CODE = {'red': 0, 'green': 1, 'blue': 2}


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


class BubbleKicker(object):

    def __init__(self, filename, channel='red'):

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
                          'with blocksize {} and cvalue {}'
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

    def label_bubbles(self):
        """provide a label for each bubble in the image"""

        ret, markers = cv.connectedComponents(1 - self.current_image)

        return ret, markers

    def perform_pipeline_canny(self, threshold, dilate_footprint,
                                border_buffer_size, border_bgval,
                                erode_footprint):
        """execute the different algorithms as a pipeline
        with given settings"""
        self.edge_detect_canny_opencv(threshold)
        self.dilate_opencv(dilate_footprint)
        self.fill_holes_opencv()
        self.clear_border_skimage(border_buffer_size, border_bgval)
        self.erode_opencv(erode_footprint)

        return self.current_image

    def what_have_i_done(self):
        """ print the current log statements"""
        self.logs.print_log_sequence()

    def plot(self):
        """plot the current image"""
        fig, ax = plt.subplots()
        ax.imshow(self.current_image, cmap=plt.cm.gray)
        ax.set_title(self.logs.log[-1])
        return fig, ax

    def calculate_properties(self):
        """calculate the required statistics"""
        # regionprops
        return None

    def calculate_distribution(self):
        """calculate and create the distribution plot"""
        # using the elf.current_image => calculate and derive distribution plot
        # you could opt to have the plot function itself outside the class
        # as this makes it more general
        return None

class BatchBubbleKicker(object):

    def __init__(self, folder, channel='red'):
        pass
