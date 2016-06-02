import microstructure.segment
import skimage.io
import skimage.filters
import skimage.transform
import skimage.color
import os
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

im = skimage.color.rgb2gray(skimage.io.imread('/home/bbales2/lukestuff/flat/GTD444_1229_2pct_10000x_Transverse_DendriteCore.TIF'))[:1600, :1600]

im = skimage.transform.resize(im, [400, 400])

# Perform a ChanVese segmentation
cv = microstructure.segment.EMMPM(im.shape)

seg = cv.run(im)

# Visualize the segmentation
plt.imshow(im, interpolation = 'NONE', cmap = plt.cm.gray)
plt.imshow(seg > skimage.filters.threshold_otsu(seg), alpha = 0.5, interpolation = 'NONE')
plt.show()
