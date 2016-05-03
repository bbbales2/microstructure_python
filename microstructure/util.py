import numpy
import scipy.cluster

def kmeans_rescale(im):
    vs, q = scipy.cluster.vq.kmeans(numpy.random.choice(im.flatten(), min(1000, im.shape[0] * im.shape[1])), 2)
    vs = sorted(vs)

    im = numpy.array(im) - vs[0]
    im /= vs[1] - vs[0]
    im -= 0.5
    im *= 2.0

    return im

def rescale(im, minv, maxv):
    if type(im) != numpy.ndarray:
        im = numpy.array(im)

    im -= im.flatten().min()
    im /= im.flatten().max()

    im *= maxv - minv
    im += minv

    return im
