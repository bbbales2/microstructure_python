import numpy

def rescale(im, minv, maxv):
    if type(im) != numpy.ndarray:
        im = numpy.array(im)

    im -= im.flatten().min()
    im /= im.flatten().max()

    im *= maxv - minv
    im += minv

    return im
