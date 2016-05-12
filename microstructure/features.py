import pyximport
pyximport.install()

from hog import hog

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
