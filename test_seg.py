import microstructure
import skimage.io
import skimage.transform
import skimage.color
import os
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

im = skimage.color.rgb2gray(skimage.io.imread('molybdenum1.png')[:1024, :1024])

im = skimage.transform.resize(im, [512, 512])

print im.shape

# Perform a Cahn Hilliard segmentation
ch = microstructure.segment.CahnHilliard(im.shape)

#seg = ch.run(im, max_steps = 100, y = 0.05)

ch.fitV(im, [25, 25], max_steps = 10000, dt = 0.1)

# Visualize the segmentation
#plt.imshow(im, cmap = plt.cm.gray, interpolation = 'NONE')
plt.imshow(seg, cmap = plt.cm.gray, interpolation = 'NONE')
plt.colorbar()
plt.show()

exit(0)

#We can try to learn a V parameter and do the segmentation again
ch.fitV(im)
seg = ch.run(im)

# Visualize the segmentation
plt.imshow(im, cmap = plt.cm.gray, interpolation = 'NONE')
plt.imshow(seg, alpha = 0.5, interpolation = 'NONE')
plt.show()

# Perform a ChanVese segmentation
cv = microstructure.segment.ChanVese(im.shape)

seg = cv.run(im)

# Visualize the segmentation
plt.imshow(im, cmap = plt.cm.gray, interpolation = 'NONE')
plt.imshow(seg, alpha = 0.5, interpolation = 'NONE')
plt.show()

# Perform a modified ChanVese segmentation
mcv = microstructure.segment.ModifiedChanVese(im.shape, 0.75)

seg = mcv.run(im)

# Visualize the segmentation
plt.imshow(im, cmap = plt.cm.gray, interpolation = 'NONE')
plt.imshow(seg, alpha = 0.5, interpolation = 'NONE')
plt.show()
