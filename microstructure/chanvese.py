import tensorflow as tf
import numpy
import util
import scipy

class ChanVese():
    """
    This class implements a Chan-Vese algorithm similarly to the one implemented by Pascal Getreur here: ipol.im/pub/art/2012/g-cv/article.pdf (Chan-Vese Segmentation).

    I didn't get the nu parameter easily working so I left it out.
    """
    def __init__(self, shape, lambda1 = 0.1, lambda2 = 0.1, mu = 0.1):
        GradCpu = numpy.concatenate((numpy.array([[0, 0, 0],
                                                  [-0.5, 0, 0.5],
                                                  [0, 0, 0]]).reshape((3, 3, 1, 1)),
                                     numpy.array([[0, -0.5, 0],
                                                  [0, 0, 0],
                                                  [0, 0.5, 0]]).reshape((3, 3, 1, 1))), axis = 3).astype('float32')

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
        #self.blurG = tf.nn.conv2d(self.Gv, self.kernel, strides = [1, 1, 1, 1], padding = 'SAME')
        #self.H = 0.5 * (1.0 + (2.0 / numpy.pi) * tf.tanh(self.blur))
        #self.conv = tf.nn.conv2d(self.H, self.Grad, strides = [1, 1, 1, 1], padding = 'SAME')

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
        import matplotlib.pyplot as plt

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

                if i % 100 == 0 and i > 0:
                    plt.imshow(im, cmap = plt.cm.gray)
                    plt.imshow(self.blur.eval()[0, :, :, 0], alpha = 0.5)
                    plt.colorbar()
                    plt.show()

                    import pdb
                    pdb.set_trace()

            plt.plot(losses)
            plt.show()

            output = self.blur.eval()[0, :, :, 0]

        return output
