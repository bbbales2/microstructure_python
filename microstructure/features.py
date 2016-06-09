import pyximport
pyximport.install(reload_support=True)

import hog
import skimage.filters

def hog2(im, bins = 20, stride = 1, sigma = 0.0):
    Nios = im.shape[0] / stride
    Njos = im.shape[1] / stride

    imp = numpy.pad(im, 1, mode = 'reflect').astype('float')

    if sigma > 0.0:
        imp = skimage.filters.gaussian(imp, sigma)
    
    output = numpy.zeros((Nios, Njos, bins), dtype = 'float')

    for io in range(Nios):
        i = io * stride + 1
        for jo in range(Njos):
            j = jo * stride + 1

            dy = (imp[i + 1, j] - imp[i - 1, j]) / 2.0
            dx = (imp[i, j + 1] - imp[i, j - 1]) / 2.0
            
            angle = (numpy.arctan2(dy, dx) + numpy.pi) / (2.0 * numpy.pi)
            
            b = int(min(bins - 1, numpy.floor(angle * bins)))
            
            output[io, jo, b] += numpy.sqrt(dx * dx + dy * dy)

    return output

class HOG():
    def __init__(self, binsize = 8, padding = False):
        self.binsize = binsize
        self.padding = padding

    def run(self, im):
        if self.padding:
            return hog.hogpad(hog.hog(im, self.binsize))
        else:
            return hog.hog(im, self.binsize)

from boxes import labels2boxes, hists2boxes
