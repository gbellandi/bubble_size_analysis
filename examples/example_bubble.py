
import matplotlib.pyplot as plt

from bubblekicker.bubblekicker import BubbleKicker

# setup the image handling object
import os
print(os.listdir("./"))

my_image = BubbleKicker('drafts/0325097m_0305.tif', channel='red')


#
my_image.perform_pipeline_opencv([100, 150], 3, 3, 1, 1)


# edge detect
#my_image.edge_detect_skimage(3, threshold=[None, None])
my_image.plot()
my_image.what_have_i_done()


# fig, ax = plt.subplots()
# ax.imshow(my_image.current_image, cmap=plt.cm.gray)
# ax.set_title("edges detected")
#
# # Dilate
# my_image.dilate_skimage()
#
# fig, ax = plt.subplots()
# ax.imshow(my_image.current_image, cmap=plt.cm.gray)
# ax.set_title("dilated image")


#...

plt.show()