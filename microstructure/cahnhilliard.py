import tensorflow as tf
import numpy
import util
import tempfile

class CahnHilliard():
    """This is a class for simulating a binary spinodal decomposition (useful for segmenting superalloy microstructure images) using a Cahn-Hilliard PDE on a 2D grid.

    An example use case is something like:
    import microstructure, skimage.io, matplotlib.pyplot

    im = skimage.io.imread('microstructure.png') # Read in a grayscale image

    ch = microstructure.segment.CahnHilliard(im.shape)

    seg = ch.run(im)

    matplotlib.pyplot.imshow(seg > 0.0)
    matplotlib.pyplot.show()

    The actual output of the program (seg in the example above), is *not* actually a segmented image. It's just the output of running the Cahn-Hilliard PDE on the input. But! The output should be easy to segment with some sort of threshold (Otsu, in particular).

    The Cahn-Hilliard PDE is structured as given on Wikipedia: https://en.wikipedia.org/wiki/Cahn%E2%80%93Hilliard_equation

    The equations are solved semi-implicitly with a spectral method. The diffusion coefficient is merged into the timestep parameter.

    So you have a better idea what the numerics look like, here is what the fully explicit form looks like:
    cr = numpy.real(numpy.fft.ifft2(u2))
    c = c - dt * (wx2 + wy2) * (numpy.fft.fft2(c**3 - c) + y * (wx2 + wy2) * c)

    The wx2/wy2 terms are the derivative operators in their spectral form.

    The semi-implicit form actually used here is:
    c3 = numpy.fft.fft2(numpy.real(numpy.fft.ifft2(u2))**3)
    c = c / (1 + (wx2 + wy2) * dt * (-1 + y * (wx2 + wy2)))
    c = c - (wx2 + wy2) * dt * c3

    In the Cahn-Hilliard equation, c is conserved. For segmentation applications, this is important. If you're trying to segment a y-y' microstructure that is 90% y and 10% y', this'll be a problem and you'll want to play around with the mean_shift parameter in the run function (if it's 50% y and 50% y', you're probably okay with a default mean_shift)!

    It runs on Nvidia GPUs using Google Tensorflow"""
    def __init__(self, shape, dt = 0.01, y = 0.25):
        """The shape of the Cahn-Hilliard PDE is specified at construction. This just allocates memory and builds the Tensorflow compute graph.

        Arguments:
        shape -- 2 element iterable, for instance [256, 127]. Size of image to segment. 3D simulations are not supported (required)

        dt -- Timestep of the Cahn-Hilliard solver. In terms of the equations on https://en.wikipedia.org/wiki/Cahn%E2%80%93Hilliard_equation, it's actually the timestep times the diffusion coefficient. Because we don't care about actual physical units here, we just merge em' together (default : 0.01)

        y -- Gamma parameter of the Cahn-Hilliard equation (https://en.wikipedia.org/wiki/Cahn%E2%80%93Hilliard_equation) (default : 0.25)
        """

        if len(shape) == 3:
            raise Exception("Shape must be 2 element iterable -- 3d CahnHilliard segmentation not supported yet!")

        self.shape = shape

        self.y = y
        self.dt = dt

        Wx = 2.0 * numpy.pi * numpy.fft.fftfreq(shape[1], 1.0)
        Wy = 2.0 * numpy.pi * numpy.fft.fftfreq(shape[0], 1.0)
        wx, wy = numpy.meshgrid(Wx, Wy)
        wx2, wy2 = numpy.meshgrid(Wx * Wx, Wy * Wy)

        self.wx2 = tf.Variable(wx2.astype('complex64'))
        self.wy2 = tf.Variable(wy2.astype('complex64'))

        self.xinp = tf.placeholder(tf.float32, shape = shape)
        self.zeros = tf.zeros(shape)

        self.x = tf.Variable(numpy.zeros(shape).astype('complex64'))
        self.ix = tf.real(tf.ifft2d(self.x))
        self.x3 = tf.fft2d(tf.complex(self.ix * self.ix * self.ix, self.zeros))

        self.xOld = tf.Variable(numpy.zeros(shape).astype('complex64'))
        self.saveX = self.xOld.assign(self.x)

        self.implicit = self.x / (1.0 + self.dt * (self.wx2 + self.wy2) * (-1.0 + self.y * (self.wx2 + self.wy2)))

        self.update1 = self.x.assign(self.implicit)
        self.update2 = self.x.assign_sub(self.dt * (self.wx2 + self.wy2) * self.x3)

        self.computeChange = tf.reduce_mean(tf.complex_abs(self.xOld - self.x))

        self.initialize = self.x.assign(tf.fft2d(tf.complex(self.xinp, self.zeros)))

    def run(self, im, stopping_threshold = 1e-4, max_steps = 50, mean_shift = 0.0):
        """Run the Cahn-Hilliard PDE

        Arguments:
        im -- image of size 'shape' given in constructor. The image values will be rescaled to [-1.0 + mean_shift, 1.0 + mean_shift] (required)

        stopping_threshold -- If the average per-pixel change in a time step is less than stopping_threshold, the simulation will stop. I.E.,

        xnew = xold + dx
        If dx < stopping_threshold, the code will stop.

        max_steps -- If the stopping threshold isn't hit by this number of timesteps, quit

        mean_shift -- Amount by which to shift the input image up or down. Because the Cahn-Hilliard equation is conservative, the mean of the initial conditions will be conserved. What this effects with regards to segmentations is sometimes you find out your image is a bit starved for material to segment out all the precipitates you want. Making this a larger number can help. (Practically it should be [-1.0, 1.0])

        Returns:
        Array of size shape, the c variable from the Cahn-Hilliard simulation. It should be easy to actually segment with an Otsu threshold
        """
        import matplotlib.pyplot as plt

        im = util.kmeans_rescale(im) + mean_shift

        with tf.Session() as sess:
            losses = []
            sess.run(tf.initialize_variables([self.wx2, self.wy2, self.x, self.xOld]))
            sess.run(self.initialize, feed_dict = { self.xinp : im })

            import time

            for i in range(max_steps):
                tmp = time.time()
                sess.run(self.saveX)
                sess.run(self.update1)
                sess.run(self.update2)
                print time.time() - tmp
                losses.append(sess.run(self.computeChange))

                if len(losses) > 1 and numpy.abs(losses[-2] - losses[-1]) < stopping_threshold:
                    break
                #plt.imshow(sess.run(self.ix))
                #plt.colorbar()
                #plt.show()

            #plt.plot(losses)
            #plt.show()

            output = sess.run(self.ix)

        return output
