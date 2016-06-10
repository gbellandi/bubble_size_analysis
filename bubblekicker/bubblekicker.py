"""
S. Van Hoey
2016-06-06
"""

from skimage.data import imread
from skimage.feature import canny
from skimage.morphology import erosion, dilation, rectangle


CHANNEL_CODE = {'red': 0, 'green': 1, 'blue': 2}


class BubbleKicker(object):

    def __init__(self, filename, channel='red'):

        self.raw_file = self._read_image(filename)

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

    @staticmethod
    def _channel_control(channel):
        """check if channel is either red, green, blue"""
        if channel not in ['red', 'green', 'blue']:
            raise Exception('Not a valid channel for RGB color scheme!')

    def edge_detect_image(self, threshold):
        """perform the edge detection algorithm of Canny on the image"""

        # perform algorithm
        image = canny(self.current_image,
                      low_threshold=threshold[0],
                      high_threshold=threshold[1])

        self.current_image = image
        return image

    def dilate_image(self):
        """perform the dilation of the image"""

        # set up structuring element
        # (@Giacomo, is (1, 90) and (1, 0) different? using rectangle here...
        struct_env = rectangle(1, 1)

        # perform algorithm with given environment,
        # store in same memory location
        image = dilation(self.current_image, selem=struct_env,
                         out=self.current_image)

        return image

    def fill_holes_image(self):
        """fill the holes of the image"""

        # perform algorithm
        # ...
        image = None

        # update current image
        self.current_image = image
        return image

    def clear_border_image(self):
        """clear the borders of the image"""

        # perform algorithm
        # ...erosion
        image = None

        # update current image
        self.current_image = image
        return image

    def erode_image(self):
        """erode the image"""

        # perform algorithm
        # ...
        image = None

        # update current image
        self.current_image = image
        return image

    def label_bubbles(self):
        """provide a label for each bubble in the image"""

        # perform algorithm
        # ...
        image = None

        # update current image
        self.current_image = image
        return image

    def calculate_properties(self):
        """calculate the required statistics"""
        # regionprops
        return None

    def perform_pipeline(self, threshold):
        """execute the different algorithms as a pipeline
        with given settings"""
        self.edge_detect_image(threshold)
        self.dilate_image()
        # ...

        return self.current_image