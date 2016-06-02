import microstructure.segment
import skimage.io
import skimage.filters
import skimage.transform
import skimage.color
import os
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

im = skimage.color.rgb2gray(skimage.io.imread('/home/bbales2/lukestuff/flat/ReneN5_1238_1079_2pct_10000x_Transverse_DendriteCore.TIF'))[:800, :800]

im = skimage.transform.resize(im, [400, 400])

# Perform a Cahn Hilliard segmentation
ch = microstructure.segment.CahnHilliard(im.shape, y = 1.0, dt = 0.01)

seg1 = ch.run(im, max_steps = 50, stopping_threshold = 1e-5)

# Perform a Cahn Hilliard segmentation with a V fitting step
ch = microstructure.segment.CahnHilliardWithV(im.shape, [7, 7], y = 5.0, dt = 0.01)
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
