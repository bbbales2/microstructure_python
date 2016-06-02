import microstructure.segment
import skimage.io
import skimage.transform
import skimage.filters
import os
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

im = skimage.io.imread('/home/bbales2/lukestuff/flat/GTD444_1229_2pct_10000x_Transverse_DendriteCore.TIF', as_grey = True)[:1600, :1600]

im = skimage.transform.resize(im, [400, 400])

# Perform a Cahn Hilliard segmentation
ch = microstructure.segment.CahnHilliard(im.shape, y = 1.0, dt = 0.01)

seg1 = ch.run(im, max_steps = 50, stopping_threshold = 1e-5)

# Visualize the segmentation
plt.imshow(im, cmap = plt.cm.gray, interpolation = 'NONE')
plt.imshow(seg1 > skimage.filters.threshold_otsu(seg1), alpha = 0.5, cmap = plt.cm.jet, interpolation = 'NONE')
plt.colorbar()
plt.show()
