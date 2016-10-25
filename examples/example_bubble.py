
import matplotlib.pyplot as plt

from bubblekicker.bubblekicker import BubbleKicker

###############
# EXAMPLE 1: pipeline ass such
###############

bubbler = BubbleKicker('drafts/0325097m_0305.tif', channel='red')

bubbler.perform_pipeline_canny([120, 180], 3, 3, 1, 1)

# show the resulting image of the detected bubbles
bubbler.plot()

# show the individual steps performed to get this result
bubbler.what_have_i_done()

###############
# EXAMPLE 2: individual dequence
###############

# setup the object
bubbler = BubbleKicker('drafts/0325097m_0305.tif', channel='red')
# using functions (both opencv as skimage are available)
bubbler.edge_detect_canny_opencv([30, 80])
bubbler.dilate_opencv(3)
# show the resulting image of the detected bubbles
bubbler.plot()
# show the individual steps performed to get this result
bubbler.what_have_i_done()

# retry another sequence => reset the image
bubbler.reset_to_raw()

# some alternative settings
bubbler.edge_detect_canny_opencv([100, 150])
bubbler.dilate_opencv(3)
bubbler.clear_border_skimage(3, 1)
# show the resulting image of the detected bubbles
bubbler.plot()
# show the individual steps performed to get this result
# this is the list since the reset to raw
bubbler.what_have_i_done()

bubbler.reset_to_raw()
bubbler.adaptive_threshold_opencv()
bubbler.clear_border_skimage()
bubbler.plot()
bubbler.what_have_i_done()

######################
# EXAMPLE 3
######################

from bubblekicker.pipelines import CannyPipeline

bubbler = CannyPipeline('drafts/0325097m_0305.tif', channel='red')
result = bubbler.run([120, 180], 3, 3, 1, 1)
bubbler.plot()


plt.show()

