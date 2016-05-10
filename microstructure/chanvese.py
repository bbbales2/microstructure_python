import tensorflow as tf
import numpy
import util
import scipy

class ChanVese():
    """
    This class implements a Chan-Vese algorithm similarly to the one implemented by Pascal Getreur here: ipol.im/pub/art/2012/g-cv/article.pdf (Chan-Vese Segmentation).

    I didn't get the nu parameter easily working so I left it out.

    The ChanVese algorithm tries to build a binary piecewise constant approximation to an image. This is just another way of sayinga binary segmentation.

    The Chan-Vese algorithm assumes that the image is composed of two types of material (type 1 and type 2 for all intensive purposes). It tries to label everything in the image as one of these two classes while minimizing the length of the boundary between the two.

    There are weights to control:
    1. The cost of labeling pixels type 1
    2. The cost of labeling pixels type 2
    3. The cost of having a boundary

    All this can be set in the constructor
    """
    def __init__(self, shape, lambda1 = 0.1, lambda2 = 0.1, mu = 0.1):
        """Initialize the ChanVese segmenter

        Arguments:
        shape (required) -- size of the image to segment

        lambda1 (default : 0.1) -- The cost of labeling pixels type 1 (check the Class docstring). This argument (as well as lambda2) can be used if the segmentation should be biased in one direction or the other. It's not deterministic what bits of the image get labeled with either lambda though -- this (as well as lambda2) will likely be a bit of a guess and check parameter.

        lambda2 (default : 0.1) -- The cost of labeling pixels type 2 (check the Class docstring)

        mu (default : 0.1) -- This is the cost of having a boundary. A higher value will mean less boundaries
        """
        xs = range(3)
        ys = range(3)
        Xs, Ys = numpy.meshgrid(xs, ys)
        Rs = numpy.sqrt((Xs - 1.0)**2 + (Ys - 1.0)**2)

        kernelBlurCpu = numpy.exp(-Rs / (2.0 * 0.75**2)).astype('float32')
        kernelBlurCpu /= numpy.linalg.norm(kernelBlurCpu.flatten())
        
        self.Grad = tf.constant(GradCpu)
        self.kernel = tf.constant(kernelBlurCpu.reshape([3, 3, 1, 1]))

        self.I = tf.Variable(tf.truncated_normal(shape = [1, shape[0], shape[1], 1], mean = 0.0, stddev = 0.1))
        
        self.u1 = tf.Variable(1.0)
        self.u2 = tf.Variable(-1.0)

        self.G = tf.placeholder(tf.float32, shape = shape)

        self.Gv = tf.Variable(numpy.zeros([1, shape[0], shape[1], 1]).astype('float32'))
        self.initialize = self.Gv.assign(tf.reshape(self.G, shape = [1, shape[0], shape[1], 1]))
        self.initialize2 = self.I.assign(tf.reshape(self.G, shape = [1, shape[0], shape[1], 1]))

        self.blur = tf.nn.conv2d(self.I, self.kernel, strides = [1, 1, 1, 1], padding = 'SAME')

        self.Gv = tf.Variable(numpy.zeros([1, shape[0], shape[1], 1]).astype('float32'))

        self.u1m = tf.abs(self.blur - self.u1)
        self.u2m = tf.abs(self.blur - self.u2)

        ones = numpy.ones((1, shape[0], shape[1], 1)).astype('float32')
        zeros = numpy.zeros((1, shape[0], shape[1], 1)).astype('float32')

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.mu = mu

        eta = 0.1
        self.conv = eta / (numpy.pi * (eta**2 + self.blur**2))

        self.u1t = self.lambda1 * tf.reduce_sum(tf.select(self.u2m > self.u1m, (self.Gv - self.u1)**2, zeros))
        self.u2t = self.lambda2 * tf.reduce_sum(tf.select(self.u2m <= self.u1m, (self.Gv - self.u2)**2, zeros))

        self.edgeLoss = self.mu * tf.reduce_sum(tf.abs(self.conv))

        self.loss = self.u1t + self.u2t + self.edgeLoss

        self.shape = shape

        self.train_step = tf.train.AdamOptimizer(1e-2).minimize(self.loss, var_list = [self.I, self.u1, self.u2])

    def run(self, im, stopping_threshold = 1e-4, max_steps = 250, percentile = 50.0):
        """Run the Chan-Vese segmentation

        im (required) -- image of size 'shape'

        stopping_threshold (default : 1e-4) -- If the average per-pixel change in a time step is less than stopping_threshold, the simulation will stop. I.E.,

        xnew = xold + dx
        If dx < stopping_threshold, the code will stop.

        max_steps (default : 50) -- If the stopping threshold isn't hit by this number of timesteps, quit

        percentile (default : 50.0) -- The values of the image will be shifted before running the segmentation so that the xth percentile pixel value is realigned to zero. Use this if there should be some bias in the segmentation one way or another (there's more material of one type than another). I'm not sure why this works based on how the algorithm runs, but it seems to. I probably have a bug somewhere.

        Returns:
        array
        
        array -- This is the 'phi' from the Chan-Vese algorithm. It is an array of size shape. To produce the segmentation, look at values above and below zero.
        """
        #import matplotlib.pyplot as plt

        im -= im.flatten().min()
        im /= im.flatten().max()

        targetQuartile = numpy.percentile(im.flatten(), percentile)

        im -= targetQuartile

        im *= 2.0

        #im = util.kmeans_rescale(im) + mean_shift

        #plt.hist((im).flatten(), bins = 100)
        #plt.show()

        with tf.Session() as sess:
            losses = []
            sess.run(tf.initialize_all_variables())
            sess.run(self.initialize, feed_dict = { self.G : im })
            sess.run(self.initialize2, feed_dict = { self.G : im })

            import time

            for i in range(max_steps):
                train, u1v, u2v, u1t, u2t, edgeLoss = sess.run([self.train_step, self.u1, self.u2, self.u1t, self.u2t, self.edgeLoss])

                if len(losses) > 1 and numpy.abs(losses[-2] - losses[-1]) < stopping_threshold:
                    break

                #if i % 100 == 0 and i > 0:
                #    plt.imshow(im, cmap = plt.cm.gray)
                #    plt.imshow(self.blur.eval()[0, :, :, 0], alpha = 0.5)
                #    plt.colorbar()
                #    plt.show()

                #    import pdb
                #    pdb.set_trace()

            #plt.plot(losses)
            #plt.show()

            output = self.blur.eval()[0, :, :, 0]

        return output
