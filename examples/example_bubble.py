import matplotlib.pyplot as plt

from bubblekicker.bubblekicker import (BubbleKicker, batchbubblekicker,
                                       bubble_properties_calculate,
                                       _bubble_properties_filter,
                                       bubble_properties_plot)

from bubblekicker.pipelines import CannyPipeline, AdaptiveThresholdPipeline

###############
# EXAMPLE 1: pipeline ass such
###############

# CANNY PIPELINE
bubbler = CannyPipeline('drafts/0325097m_0305.tif', channel='red')
result = bubbler.run([120, 180], 3, 3, 1, 1)
# show the resulting image of the detected bubbles
bubbler.plot()
# show the individual steps performed to get this result
bubbler.what_have_i_done()

# ADAPTIVE THRESHOLD PIPELINE
bubbler = AdaptiveThresholdPipeline('drafts/0325097m_0305.tif', channel='red')
result = bubbler.run(91, 18, 3, 1, 1)
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

# switch color channel
bubbler = BubbleKicker('drafts/0325097m_0305.tif', channel='red')
print(bubbler.what_channel())
bubbler.plot()

bubbler.switch_channel('green')
print(bubbler.what_channel())
bubbler.plot()

###############
# EXAMPLE 3: running a batch sequence
###############

res = batchbubblekicker('examples/data', 'red',
                        AdaptiveThresholdPipeline,
                        91, 18, 3, 1, 1)
print(res)

###############
# EXAMPLE 4: Property functions
###############

# derive and PLOT the bubble properties as a table with no filter
bubbler = CannyPipeline('drafts/0325097m_0305.tif', channel='red')
result = bubbler.run([120, 180], 3, 3, 1, 1)
id_image, props = bubble_properties_calculate(result, rules={})
fig, axs = bubble_properties_plot(props, "equivalent_diameter")
fig.savefig("examples/output_eq_diameter.png")
fig, axs = bubble_properties_plot(props, "area")
fig.savefig("examples/output_area.png")

# filter bubble properties based on a DEFAULT filter
bubbler = CannyPipeline('drafts/0325097m_0305.tif', channel='red')
result = bubbler.run([120, 180], 3, 3, 1, 1)
id_image, props = bubble_properties_calculate(result)
print(props)

# filter bubble properties based on CUSTOM filter ruleset
custom_filter = {'circularity_reciprocal': {'min': 0.2, 'max': 1.6},
                 'convexity': {'min': 1.92}}
bubbler = CannyPipeline('drafts/0325097m_0305.tif', channel='red')
result = bubbler.run([120, 180], 3, 3, 1, 1)
id_image, props = bubble_properties_calculate(result, rules=custom_filter)
print(props)

plt.show()