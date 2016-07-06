
import matplotlib.pyplot as plt

from bubblekicker import BubbleKicker

# setup the image handling object
import os
print(os.listdir("./"))

my_image = BubbleKicker('Bubble_test_053.bmp', channel='red')

# edge detect
my_image.edge_detect_image(3, threshold=[None, None])

fig, ax = plt.subplots()
ax.imshow(my_image.current_image, cmap=plt.cm.gray)
ax.set_title("edges detected")

# Dilate
my_image.dilate_image()

fig, ax = plt.subplots()
ax.imshow(my_image.current_image, cmap=plt.cm.gray)
ax.set_title("dilated image")


#...

plt.show()