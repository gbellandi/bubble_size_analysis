
from bubblekicker.bubblekicker import BubbleKicker


class CannyPipeline(BubbleKicker):

    def __init__(self, filename, channel='red'):
        super(CannyPipeline, self).__init__(filename, channel=channel)

    def run(self, threshold, dilate_footprint, border_buffer_size,
            border_bgval, erode_footprint):
		
        """Execute the different algorithms as a pipeline
        with given settings
		
        Parameters
        ----------
        threshold: [n, m]
            array of two parameters representing min and max 
            thresholds for the hysteresis procedure of the opencv
            Canny method
        dilate_footprint: int
            footprint (kernel) of the opencv dilate function.
            Should be an odd number.
        border_buffer_size: int
            width of the border around the image used to clear 
            possible partial objects
        border_bgval: int
            value to be given to the border touching objects
        
        """

        self.edge_detect_canny_opencv(threshold)
        self.dilate_opencv(dilate_footprint)
        self.fill_holes_opencv()
        self.clear_border_skimage(border_buffer_size, border_bgval)
        self.erode_opencv(erode_footprint)

        return self.current_image


class AdaptiveThresholdPipeline(BubbleKicker):

    def __init__(self, filename, channel='red'):
        super(AdaptiveThresholdPipeline, self).__init__(filename,
                                                        channel=channel)

    def run(self, blocksize, cvalue, border_buffer_size,
            border_bgval, erode_footprint):
        """execute the different algorithms as a pipeline
        with given settings"""
        self.adaptive_threshold_opencv(blocksize, cvalue)
        self.clear_border_skimage(border_buffer_size, border_bgval)
        self.erode_opencv(erode_footprint)

        return self.current_image
