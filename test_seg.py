import microstructure
import skimage.io
import skimage.filters
import skimage.transform
import skimage.color
import os
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

#im = skimage.color.rgb2gray(skimage.io.imread('/home/bbales2/rafting/nrafting2a/images_9/signal0.png'))[:256, :256]
#im = skimage.color.rgb2gray(skimage.io.imread('/home/bbales2/lukestuff/flat/ReneN5_1238_1079_2pct_10000x_Transverse_DendriteCore.TIF'))[:800, :800]
#im = skimage.color.rgb2gray(skimage.io.imread('/home/bbales2/lukestuff/flat/GTD444_1229_0pct_10000x_Transverse_DendriteCore.TIF'))[:800, :800]
#im = skimage.color.rgb2gray(skimage.io.imread('/home/bbales2/lukestuff/flat/GTD444_1245_1080_0pct_10000x_Transverse_Interdendritic.TIF'))[:1600, :1600]
im = skimage.color.rgb2gray(skimage.io.imread('/home/bbales2/lukestuff/flat/GTD444_1229_2pct_10000x_Transverse_DendriteCore.TIF'))[:1600, :1600]
#im = skimage.color.rgb2gray(skimage.io.imread('/home/bbales2/DeepTextures/Images/reneN5.png'))
#im = skimage.color.rgb2gray(skimage.io.imread('molybdenum1.png'))[:1024, :1024]

im = skimage.transform.resize(im, [256, 256])

print im.shape

# Perform a ChanVese segmentation
cv = microstructure.segment.ChanVese(im.shape)

seg = cv.run(im, percentile = 50.0, max_steps = 1000)

# Visualize the segmentation
plt.imshow(seg, interpolation = 'NONE', cmap = plt.cm.gray)
plt.imshow(seg > skimage.filters.threshold_otsu(seg), alpha = 0.5, interpolation = 'NONE')
#plt.imshow(im, cmap = plt.cm.gray, interpolation = 'NONE')
plt.show()
exit(0)
# Perform a Cahn Hilliard segmentation
ch = microstructure.segment.CahnHilliard(im.shape, y = 1.0, dt = 0.05)

seg1 = ch.run(im, max_steps = 50, stopping_threshold = 1e-5)

# Visualize the segmentation
#plt.imshow(im, cmap = plt.cm.gray, interpolation = 'NONE')
#plt.imshow(seg1, cmap = plt.cm.jet, interpolation = 'NONE')
#plt.colorbar()
#plt.show()

# Perform a Cahn Hilliard segmentation
ch = microstructure.segment.CahnHilliardWithV(im.shape, [7, 7], y = 1.0, dt = 0.05)
V = ch.fit(im, max_steps = 10000, stopping_threshold = 1e-6)

seg2 = ch.run(im, max_steps = 50, stopping_threshold = 1e-5)

plt.imshow(V, interpolation = 'NONE')
plt.colorbar()
plt.show()

# Visualize the segmentation
plt.subplot(1, 3, 0)
plt.imshow(im, cmap = plt.cm.jet, interpolation = 'NONE')
plt.title('Original')
#plt.colorbar()
plt.subplot(1, 3, 1)
plt.imshow(seg1, cmap = plt.cm.jet, interpolation = 'NONE')
plt.title('Seg 1')
#plt.colorbar()
plt.subplot(1, 3, 2)
plt.imshow(seg2, cmap = plt.cm.jet, interpolation = 'NONE')
plt.title('Seg 2')
#plt.colorbar()
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
