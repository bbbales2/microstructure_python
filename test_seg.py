import microstructure
import skimage.io
import skimage.transform
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

im = skimage.io.imread('molybdenum1.png')[:1024, :1024]

im = skimage.transform.resize(im, [64, 64])

# Perform a Cahn Hilliard segmentation
ch = microstructure.segment.CahnHilliard(im.shape)

seg = ch.run(im)

# Visualize the segmentation
plt.imshow(im, cmap = plt.cm.gray, interpolation = 'NONE')
plt.imshow(seg, alpha = 0.5, interpolation = 'NONE')
plt.show()

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
