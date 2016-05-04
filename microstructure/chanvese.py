import tensorflow as tf
import numpy
import util
import scipy

class ChanVese():
    def __init__(self, shape, lambda1 = 0.1, lambda2 = 0.1, nu = 0.0, mu = 0.1):
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
        self.nu = nu
        self.mu = mu

        eta = 0.1
        self.conv = eta / (numpy.pi * (eta**2 + self.blur**2))

        self.areaLoss = self.nu * tf.reduce_sum(tf.select(self.u1m < self.u2m, ones, zeros))
        self.u1t = self.lambda1 * tf.reduce_sum(tf.select(self.u2m > self.u1m, (self.Gv - self.u1)**2, zeros))
        self.u2t = self.lambda2 * tf.reduce_sum(tf.select(self.u2m <= self.u1m, (self.Gv - self.u2)**2, zeros))

        self.edgeLoss = self.mu * tf.reduce_sum(tf.abs(self.conv))

        self.loss = self.u1t + self.u2t + self.areaLoss + self.edgeLoss

        self.shape = shape

        self.train_step = tf.train.AdamOptimizer(1e-2).minimize(self.loss, var_list = [self.I, self.u1, self.u2])

    def run(self, im, stopping_threshold = 1e-4, max_steps = 250, percentile = 50.0):
        import matplotlib.pyplot as plt

        #im -= im.flatten().mean()
        #im /= im.flatten().std()

        im -= im.flatten().min()
        im /= im.flatten().max()

        targetQuartile = numpy.percentile(im.flatten(), percentile)

        im -= targetQuartile#0.5

        im *= 2.0

        #im = util.kmeans_rescale(im) + mean_shift

        #plt.hist((im).flatten(), bins = 100)
        #plt.show()

        with tf.Session() as sess:
            losses = []
            sess.run(tf.initialize_all_variables())
            sess.run(self.initialize, feed_dict = { self.G : im })
            sess.run(self.initialize2, feed_dict = { self.G : im })

            #u1a = numpy.median(im[im < 0.0].flatten())
            #u2a = numpy.median(im[im >= 0.0].flatten())

            #u3a = (abs(u1a) + u2a) / 2.0
            #sess.run(self.u1.assign(-u3a))
            #sess.run(self.u2.assign(u3a))

            print sess.run(self.u1)
            print sess.run(self.u2)

            #import pdb
            #pdb.set_trace()

            import time

            for i in range(max_steps):
                tmp = time.time()
                train, u1v, u2v, u1t, u2t, edgeLoss, areaLoss = sess.run([self.train_step, self.u1, self.u2, self.u1t, self.u2t, self.edgeLoss, self.areaLoss])
                print time.time() - tmp
                #losses.append(loss)

                print u1v, u2v, u1t, u2t, edgeLoss, areaLoss

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

class NotQuiteChanVese():
    def __init__(self, shape):
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
        self.I2 = tf.Variable(tf.truncated_normal(shape = [1, shape[0], shape[1], 1], mean = 0.0, stddev = 0.1))
        
        self.u1 = tf.Variable(-1.0)
        self.u2 = tf.Variable(1.0)

        self.G = tf.placeholder(tf.float32, shape = shape)

        self.Gv = tf.Variable(numpy.zeros([1, shape[0], shape[1], 1]).astype('float32'))
        self.initialize = self.Gv.assign(tf.reshape(self.G, shape = [1, shape[0], shape[1], 1]))

        self.blur = tf.nn.conv2d(self.I, self.kernel, strides = [1, 1, 1, 1], padding = 'SAME')
        self.blurG = tf.nn.conv2d(self.Gv, self.kernel, strides = [1, 1, 1, 1], padding = 'SAME')
        self.conv = tf.nn.conv2d(self.blur, self.Grad, strides = [1, 1, 1, 1], padding = 'SAME')

        Eloss = 0.1 * tf.nn.l2_loss(tf.minimum(tf.abs(self.blur - self.u1), tf.abs(self.blur - self.u2)))

        Gloss = tf.nn.l2_loss(self.blur + self.I2 - tf.reshape(self.G, [1, shape[0], shape[1], 1]))
        Wloss = 1.0 * tf.nn.l2_loss(self.I2)
        Rloss = 0.05 * tf.reduce_sum(tf.abs(self.conv))
        self.err = (tf.reduce_mean(self.blur) - self.u1) / (self.u2 - self.u1)
        #Tloss = 5000.0 * (self.err - 0.75)**2

        self.loss = Eloss + Gloss + Rloss + Wloss# + Tloss

        self.shape = shape

        self.train_step = tf.train.AdamOptimizer(1e-2).minimize(self.loss, var_list = [self.I, self.I2])#, u1, u2

    def run(self, im, stopping_threshold = 1e-4, max_steps = 250, percentile = 50.0):
        import matplotlib.pyplot as plt

        #im -= im.flatten().mean()
        #im /= im.flatten().std()

        im -= im.flatten().min()
        im /= im.flatten().max()

        targetQuartile = numpy.percentile(im.flatten(), percentile)

        im -= targetQuartile#0.5

        im *= 2.0

        #im = util.kmeans_rescale(im) + mean_shift

        plt.hist((im).flatten(), bins = 100)
        plt.show()

        with tf.Session() as sess:
            losses = []
            sess.run(tf.initialize_all_variables())
            sess.run(self.initialize, feed_dict = { self.G : im })

            #u1a = numpy.median(im[im < 0.0].flatten())
            #u2a = numpy.median(im[im >= 0.0].flatten())

            #u3a = (abs(u1a) + u2a) / 2.0
            #sess.run(self.u1.assign(-u3a))
            #sess.run(self.u2.assign(u3a))

            print sess.run(self.u1)
            print sess.run(self.u2)

            #import pdb
            #pdb.set_trace()

            import time

            for i in range(max_steps):
                tmp = time.time()
                train, loss = sess.run([self.train_step, self.loss], feed_dict = { self.G : im })
                print time.time() - tmp
                losses.append(loss)

                if len(losses) > 1 and numpy.abs(losses[-2] - losses[-1]) < stopping_threshold:
                    break

                #if i % 250 == 0 and i > 0:
                    #import pdb
                    #pdb.set_trace()

                    #plt.imshow(self.blur.eval()[0, :, :, 0])
                    #plt.colorbar()
                    #plt.show()

            plt.plot(losses)
            plt.show()

            output = self.blur.eval()[0, :, :, 0]

        return output
