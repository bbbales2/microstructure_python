import numpy

def rescale(im, minv, maxv):
    im = numpy.array(im)

    im -= im.flatten().min()
    im /= im.flatten().max()

    im *= maxv - minv
    im -= minv
