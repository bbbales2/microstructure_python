#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

cimport numpy
import numpy

print 'hi'

cpdef labels2boxes(labels_, int Nlabels, int size = 9, int stride = 8, padding_mode = None):
    cdef numpy.ndarray[numpy.int_t, ndim = 2] labels = labels_.astype('int')
    cdef numpy.ndarray[numpy.float_t, ndim = 3] output

    cdef int i, j, ii, jj, io, jo, Nios, Njos

    Nios = labels.shape[0] / stride
    Njos = labels.shape[1] / stride

    if padding_mode:
        labels = numpy.pad(labels, size / 2, mode = padding_mode)

    output = numpy.zeros((Nios, Njos, Nlabels)).astype('float')

    for io in range(Nios):
        i = io * stride + size / 2
        for jo in range(Njos):
            j = jo * stride + size / 2
            for ii in range(i - size / 2, i + (size + 1) / 2):
                for jj in range(j - size / 2, j + (size + 1) / 2):
                    output[io, jo, labels[ii, jj]] += 1.0

    return output / float(size * size)

cpdef hog2boxes(hists_, int b = 9, padding_mode = None):
    cdef numpy.ndarray[numpy.float_t, ndim = 3] hists = hists_.astype('float')
    cdef numpy.ndarray[numpy.float_t, ndim = 3] output

    cdef int i, j, ii, jj, c

    if padding_mode:
        hists = numpy.pad(hists, ((b / 2, b / 2), (b / 2, b / 2), (0, 0)), mode = padding_mode)

    output = numpy.zeros((hists.shape[0] - (b / 2) * 2, hists.shape[1] - (b / 2) * 2, hists.shape[2]))

    for i in range(b / 2, hists.shape[0] - b / 2):
        for j in range(b / 2, hists.shape[1] - b / 2):
            for ii in range(i - b / 2, i + b / 2 + 1):
                for jj in range(j - b / 2, j + b / 2 + 1):
                    for c in range(hists.shape[2]):
                        output[i - b / 2, j - b / 2, c] += hists[ii, jj, c]

    output /= float(b * b)

    return output

cpdef hists2boxes(hists_, int b = 9, padding_mode = None):
    cdef numpy.ndarray[numpy.float_t, ndim = 3] hists = hists_.astype('float')
    cdef numpy.ndarray[numpy.float_t, ndim = 3] output

    cdef int i, j, ii, jj, c

    if padding_mode:
        hists = numpy.pad(hists, ((b / 2, b / 2), (b / 2, b / 2), (0, 0)), mode = padding_mode)

    output = numpy.zeros((hists.shape[0] - (b / 2) * 2, hists.shape[1] - (b / 2) * 2, hists.shape[2]))

    for i in range(b / 2, hists.shape[0] - b / 2):
        for j in range(b / 2, hists.shape[1] - b / 2):
            for ii in range(i - b / 2, i + b / 2 + 1):
                for jj in range(j - b / 2, j + b / 2 + 1):
                    for c in range(hists.shape[2]):
                        output[i - b / 2, j - b / 2, c] += hists[ii, jj, c]

    return output
