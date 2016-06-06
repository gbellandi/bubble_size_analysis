
import matplotlib.pyplot as plt

from bubblekicker import BubbleKicker

# setup the image handling object
my_image = BubbleKicker('Bubble test_053.bmp', channel='red')

# edge detect
my_image.edge_detect_image([0.01, 0.5])

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