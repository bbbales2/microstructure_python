import microstructure.features
import skimage.io
import skimage.color
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

im = skimage.color.rgb2gray(skimage.io.imread('molybdenum1.png'))

# Global features
hog = microstructure.features.HOG(8)

hogs = hog.run(im)

1/0
# Sparse features?

# Dense features, lbps, HOGs

cifar = microstructure.features.Cifar10CNN(shape = [16, 16], stride = [8, 8], pad = 'edge')

feats = cifar.run(im)

kmeans = microstructure.classifiers.Kmeans(3)

feats3 = kmeans.fit_predict(feats)

# Smooth out the features a bit

cv = microstructure.segment.ChanVese(3, separate_channels = True)

feats3_smooth = cv.run(feats3)

plt.imshow(im, cmap = plt.cm.gray, interpolation = 'NONE')
plt.imshow(feats3_smooth, alpha = 0.5, interpolation = 'NONE')
plt.show()

# HOG features

hog = microstructure.features.HOG(sigma = 1.0, shape = [16, 16], stride = [8, 8], normalize = False)

hogs = hog.run(im)

cv = microstructure.segment.ChanVese(3, separate_channels = True)

feats3_smooth = cv.run(hogs)

plt.imshow(im, cmap = plt.cm.gray, interpolation = 'NONE')
plt.imshow(feats3_smooth, alpha = 0.5, interpolation = 'NONE')
plt.show()
