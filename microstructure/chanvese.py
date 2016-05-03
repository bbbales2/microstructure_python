import tensorflow as tf
import numpy
import util
import scipy

class ChanVese():
    def __init__(self, shape):
        GradCpu = numpy.concatenate((numpy.array([[0, 0, 0],
                                                  [-0.5, 0, 0.5],
                                                  [0, 0, 0]]).reshape((3, 3, 1, 1)),
                                     numpy.array([[0, -0.5, 0],
                                                  [0, 0, 0],
                                                  [0, 0.5, 0]]).reshape((3, 3, 1, 1))), axis = 3).astype('float32')
        
        kernelCpu = scipy.ndimage.filters.gaussian_filter([[0.0, 0.0, 0.0],
                                                           [0.0, 1.0, 0.0],
                                                           [0.0, 0.0, 0.0]], 0.75).reshape((3, 3, 1, 1)).astype('float32')
        
        kernelCpu /= kernelCpu.flatten().sum()
        
        self.Grad = tf.constant(GradCpu)
        self.kernel = tf.constant(kernelCpu)
        
        self.I = tf.Variable(tf.truncated_normal(shape = [1, shape[0], shape[1], 1], mean = 0.0, stddev = 0.1))
        self.I2 = tf.Variable(tf.truncated_normal(shape = [1, shape[0], shape[1], 1], mean = 0.0, stddev = 0.1))
        
        self.u1 = tf.Variable(-1.0)
        self.u2 = tf.Variable(1.0)

        self.G = tf.placeholder(tf.float32, shape = shape)

        self.Gv = tf.reshape(self.G, shape = [1, shape[0], shape[1], 1])

        self.blur = tf.nn.conv2d(self.I, self.kernel, strides = [1, 1, 1, 1], padding = 'SAME')
        self.blurG = tf.nn.conv2d(self.Gv, self.kernel, strides = [1, 1, 1, 1], padding = 'SAME')
        self.conv = tf.nn.conv2d(self.blur, self.Grad, strides = [1, 1, 1, 1], padding = 'SAME')

        Eloss = 0.1 * tf.nn.l2_loss(tf.minimum(tf.abs(self.blur - self.u1), tf.abs(self.blur - self.u2)))

        Gloss = tf.nn.l2_loss(self.blur + self.I2 - tf.reshape(self.G, [1, shape[0], shape[1], 1]))
        Wloss = 1.0 * tf.nn.l2_loss(self.I2)
        Rloss = 0.05 * tf.reduce_sum(tf.abs(self.conv))
        self.err = (tf.reduce_mean(self.blur) - self.u1) / (self.u2 - self.u1)
        Tloss = 5000.0 * (self.err - 0.5)**2

        self.loss = Eloss + Gloss + Rloss + Wloss + Tloss

        self.shape = shape

        self.train_step = tf.train.AdamOptimizer(1e-2).minimize(self.loss, var_list = [self.I, self.I2])#, u1, u2

    def run(self, im, stopping_threshold = 1e-5, max_steps = 50, mean_shift = 0.0):
        import matplotlib.pyplot as plt

        #im -= im.flatten().min()
        #im /= im.flatten().max()
        #im -= numpy.mean(im.flatten())#0.5
        #im *= 2.0

        im = util.kmeans_rescale(im) + mean_shift

        plt.hist(im.flatten(), bins = 100)
        plt.show()

        with tf.Session() as sess:
            losses = []
            sess.run(tf.initialize_all_variables())
            #sess.run(self.initialize, feed_dict = { self.xinp : im })

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
