#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy
cimport numpy
cimport libc.math
import skimage.filters

class EMMPM():
    """This is an implementation of the EMMPM segmentation algorithm.

    See 'The EM/MPM Algorithm for Segmentation of Textured Images: Analysis and Further Experimental Results' by Mary Comer and Edward Delp for more information on the math.

    In this implementation the number of classes is assumed to be two.
    """
    def __init__(self, shape, interactionEnergy = 5.0):
        """Initialize the EMMPM segmenter.

        Arguments:
        shape (required) -- size of image to segment

        interactionEnergy (default: 5.0) -- This is the value of the interaction parameter in the EMMPM paper. A larger value means that neighboring pixels are more strongly encouraged to take the same value
        """
        self.H = shape[0]
        self.W = shape[1]
        self.b = interactionEnergy

    def run(self, image, mpmSamples_ = 10, iterations_ = 10):
        """Run the EMMPM segmentation.

        Basically the calculation cosists of two parts: an EM stage, and an MPM stage (if you want to know what those acronyms mean, read the paper). These are run in an alternating pattern for some number of iterations.

        The MPM stage itself is an interative calculation. It's run mpmSamples times at each iteration of the outer loop.

        So in pseudo-code this looks like:

        for i in range(iterations):
            do_em_update

            for j in range(mpmSamples_):
                do_mpm_iter_update

        Aruments:
        image (required) -- Image of size 'shape' given in the constructor.

        mpmSamples_ (default : 10) -- This is the number of Gibbs samplings that are done to compute P(X|Y, theta). Think of it as the inner loop of the computation

        iterations_ (default : 10) -- This is the number of times that the EM and then MPM updates are run. Think of this as the outer loop of the computation
        
        
        Returns:
        array

        array -- array is an array of size shape that holds the output of the Cahn-Hilliard simulations. It should be easy to actually segment with an Otsu threshold
        """
        if image.shape[0] != self.H or image.shape[1] != self.W:
            raise Exception("EMMPM passed image of shape {0} but expected shape {1}".format(image.shape, [self.H, self.W]))

        cdef int mpmSamples = <int>mpmSamples_
        cdef int iterations = <int>iterations_

        cdef double b = <double>self.b

        cdef int H = <int>self.H
        cdef int W = <int>self.W

        cdef numpy.ndarray[numpy.double_t, ndim = 2] Y = image.astype('double')
        cdef numpy.ndarray[numpy.int_t, ndim = 2] X = numpy.zeros((H, W)).astype('int')

        cdef numpy.ndarray[numpy.double_t, ndim = 2] p = numpy.zeros((H, W))
    
        cdef numpy.ndarray[numpy.int_t, ndim = 3] T = numpy.zeros((H, W, 2)).astype('int')
        cdef numpy.ndarray[numpy.int_t, ndim = 2] TT = numpy.zeros((H, W)).astype('int')

        cdef numpy.ndarray[numpy.double_t, ndim = 2] randoms

        # Generate the initial segmentation guess from an Otsu threshold
        threshold = skimage.filters.threshold_otsu(Y)

        X[Y > threshold] = 1
        
        cdef int i, j, v
        cdef int r, tmp
        cdef double[2] u = [0.0, 0.0]
        cdef double[2] sig2 = [0.0, 0.0]
        cdef double psum
        cdef int[2] counts
        cdef double p0, p1
        
        for i in range(H):
            for j in range(W):
                T[i, j, X[i, j]] = 1

        for r in range(iterations):
            for i in range(H):
                for j in range(W):
                    tmp = 0

                    for c in range(2):
                        tmp += T[i, j, c]

                    TT[i, j] = tmp

            for c in range(2):
                for i in range(H):
                    for j in range(W):
                        p[i, j] = T[i, j, c] / TT[i, j]

                psum = 0.0

                for i in range(H):
                    for j in range(W):
                        psum += p[i, j]
                        u[c] += p[i, j] * Y[i, j]

                u[c] /= psum

                for i in range(H):
                    for j in range(W):
                        sig2[c] += p[i, j] * (Y[i, j] - u[c])**2

                sig2[c] /= psum

            T = numpy.zeros((H, W, 2)).astype('int')

            for s in range(mpmSamples):
                # The paper "The EM/MPM Algorithm for Segmentation of Textured Images: Analysis and Further Experimental Results" says I should only change one pixel at a time
                #   I saw an implementation from http://www.bluequartz.net/ (EMMPM workbench) where I think
                #   they changed more than one pixel at a time (I don't think I really understood the code so I could be wrong).
                #
                #   I think changing more than one pixel is fine. You're just making bigger jumps in state space,
                #   so I'm doin' it here too.
                #
                for i in range(H):
                    for j in range(W):
                        im = 0 if i - 1 < 0 else i - 1
                        ip = i + 1 if i + 1 < H else H - 1

                        jm = 0 if j - 1 < 0 else j - 1
                        jp = j + 1 if j + 1 < W else W - 1

                        counts = [0, 0]

                        counts[X[ip, j]] += 1
                        counts[X[im, j]] += 1
                        counts[X[i, jp]] += 1
                        counts[X[i, jm]] += 1

                        p0 = libc.math.exp(-((Y[i, j] - u[X[i, j]])**2) / (2.0 * sig2[X[i, j]]) - b * counts[1 - X[i, j]])
                        p1 = libc.math.exp(-((Y[i, j] - u[1 - X[i, j]])**2) / (2.0 * sig2[1 - X[i, j]]) - b * counts[X[i, j]])

                        p[i, j] = p0 / (p0 + p1)

                randoms = numpy.random.rand(H, W)
                
                for i in range(H):
                    for j in range(W):
                        if randoms[i, j] > p[i, j]:
                            X[i, j] = 1 - X[i, j]

                        T[i, j, X[i, j]] += 1

        self.u = u
        self.sig = dict([[k, numpy.sqrt(v)] for k, v in enumerate(sig2)])

        return T[:, :, 1] / (T[:, :, 0] + T[:, :, 1]).astype('double')

