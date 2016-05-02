import tensorflow as tf
import numpy
import util

class CahnHilliard():
    def __init__(self, shape, dt = 0.01):
        if len(shape) == 3:
            raise Exception("Shape must be 2 element iterable -- 3d CahnHilliard segmentation not supported yet!")
        
        alpha = 1.0
        y = 0.25

        Wx = 2.0 * numpy.pi * numpy.fft.fftfreq(shape[1], 1.0).astype('complex64')
        Wy = 2.0 * numpy.pi * numpy.fft.fftfreq(shape[0], 1.0).astype('complex64')
        wx, wy = numpy.meshgrid(Wx, Wy)
        wx2, wy2 = numpy.meshgrid(Wx * Wx, Wy * Wy)

        self.wx2 = tf.Variable(wx2)
        self.wy2 = tf.Variable(wy2)

        self.xinp = tf.placeholder(tf.float32, shape = shape)
        self.zeros = tf.zeros(shape)

        self.x = tf.Variable(numpy.zeros(shape).astype('complex64'))
        self.ix = tf.real(tf.ifft2d(self.x))
        self.x3 = tf.fft2d(tf.complex(self.ix * self.ix * self.ix, self.zeros))

        self.update1 = self.x.assign(self.x / (alpha * (self.wx2 + self.wy2) * dt * (-1.0 + y * (self.wx2 + self.wy2))))
        self.update2 = self.x.assign_sub(alpha * (self.wx2 + self.wy2) * dt * self.x3)

        self.initialize = self.x.assign(self.xinp)

    def run(self, im):
        im = util.rescale(im, -1.0, 1.0)

        self.initialize.eval(feed_dict = { self.xinp : im })

        for i in range(1):
            self.update1.eval()
            self.update2.eval()

        return self.ix.eval()
