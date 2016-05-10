import tensorflow as tf
import numpy
import util
import tempfile

import cahnhilliard

class CahnHilliardWithV(cahnhilliard.CahnHilliard):
    """This is a basic extension to the microstructure.segment.CahnHilliard segmenter that tries to automagically take into account faceting in the microstructure.

    If the regular Cahn-Hilliard explict PDE update looks something like:

    eq 1 : dc/dt = \del (c**3 - c + y \del c)

    Then the update we're working with here looks like:

    eq 2 : dc/dt = \del (c**3 - c + y \del c + conv(c, V))

    So that last convolution term has something to do with stresses/strains/and the faceting that happens in superalloys.

    The inspiration for this came from a paper from Yunzhi Wang, "Field kinetic model and computer simulation of precipitation of L12 ordered intermetallics from f.c.c. solid solution" (http://www.sciencedirect.com/science/article/pii/S1359645498000159). It shows how to propely do the faceting and antiphase domains and stuff.

    The faceting comes in through a convolution term that we happily steal for our own uses here.

    Now! V is a fancy kernel that is a function of the physics of the system. It's very unlikely that we know the V for any given system.

    Instead, we try to estimate V by assuming that our image is a steady state solution to equation #2 (it definitely isn't, but bear with me). To solve for V, we assume we know 'y' and dc/dt = 0, so that we just just say conv(c, V) = c - c**3 - y \del c.

    We can fit V using this equation, and then just run a simulation of equation 2 w/ the V to get a segmentation of our image!
    """
    def __init__(self, shape, vShape = [13, 13], dt = 0.01, y = 0.25):
        """
        The shape of the image to segment must be specified at construction.

        The size of the kernel V can be made larger or smaller, but just remember that if it's 13x13, that's 269 parameters to fit, and that'll grow quadratically!

        Arguments:
        shape -- 2 element iterable, for instance [256, 127]. Size of image to segment. 3D simulations are not supported (required)

        vShape -- 2 element iterable, for instance [7, 7]. Size of convolution kernel that'll be fit to make the segmentation facet-y (default : [13, 13])
        
        dt -- Timestep of the Cahn-Hilliard solver. In terms of the equations on https://en.wikipedia.org/wiki/Cahn%E2%80%93Hilliard_equation, it's actually the timestep times the diffusion coefficient. Because we don't care about actual physical units here, we just merge em' together (default : 0.01)

        y -- Gamma parameter of the Cahn-Hilliard equation (https://en.wikipedia.org/wiki/Cahn%E2%80%93Hilliard_equation) (default : 0.25)

        """
        cahnhilliard.CahnHilliard.__init__(self, shape = shape, dt = dt, y = y)

        if len(shape) != 2:
            raise Exception("shape must be length 2")

        if len(vShape) != 2:
            raise Exception("vShape must be length 2")

        self.shape = shape
        self.vShape = vShape

        # Set up the network for figuring out V
        self.toFit = tf.Variable(tf.truncated_normal([1, self.shape[0], self.shape[1], 1], 0.1))
        self.V = tf.Variable(tf.truncated_normal([vShape[0], vShape[1], 1, 1], 0.1))
        self.b = tf.Variable(tf.constant(0.01, shape = [1]))
        self.xV = tf.nn.conv2d(tf.reshape(self.ix, [1, self.shape[0], self.shape[1], 1]), self.V, [1, 1, 1, 1], padding = 'SAME') + self.b
        
        self.error = tf.nn.l2_loss(self.toFit - self.xV)##tf.reduce_mean(tf.abs((self.toFit - self.xV)))

        self.train_step = tf.train.AdamOptimizer(1e-2).minimize(self.error, var_list = [self.V, self.b])

        # Modify the parent update functions to use the V
        self.fftV = tf.fft2d(tf.complex(tf.reshape(self.xV, self.shape), self.zeros))

        self.saver = tf.train.Saver({ "V" : self.V, "b" : self.b })

        self.is_fit = False

    def fit(self, im, mean_shift = 0.0, max_steps = 1000, stopping_threshold = 1e-4):
        """Fit the V for the segmentation.

        Arguments:
        im -- This is the image of interest. It must be the same shape as passed to the constructor

        stopping_threshold (default : 1e-4) -- If the average per-pixel percent change (on a [0, 1] scale) in a time step is less than stopping_threshold, the simulation will stop. I.E.,

        xnew = xold + dx
        If (dx / xold) < stopping_threshold, the code will stop.

        max_steps (default : 250) -- If the stopping threshold isn't hit by this number of timesteps, quit

        mean_shift (default : 0.0) -- Amount by which to shift the input image up or down. Because the Cahn-Hilliard equation is conservative, the mean of the initial conditions will be conserved. What this effects with regards to segmentations is sometimes you find out your image is a bit starved for material to segment out all the precipitates you want. Making this a larger number can help. (Practically it should be [-1.0, 1.0])

        Returns:
        V

        V -- V is an array of size vShape that has the computed convolution kernel V
        """
        self.is_fit = True

        import matplotlib.pyplot as plt

        im = util.kmeans_rescale(im) + mean_shift

        with tf.Session() as sess:
            losses = []
            sess.run(tf.initialize_all_variables())
            sess.run(self.initialize, feed_dict = { self.xinp : im })

            sess.run(self.saveX)
            sess.run(self.update1)
            sess.run(self.update2)

            inv = (self.dt * (self.wx2 + self.wy2)).eval()
            inv[0, 0] = 1.0

            M = ((self.x - self.xOld) / inv).eval()
            M[0, 0] = 0.0

            #import pdb
            #pdb.set_trace()

            sess.run(self.toFit.assign(tf.reshape(numpy.real(numpy.fft.ifft2(M)).astype('float32'), [1, self.shape[0], self.shape[1], 1])))

            sess.run(self.x.assign(self.xOld))

            #import pdb
            #pdb.set_trace()

            import time

            for i in range(max_steps):
                tmp = time.time()
                train, error = sess.run([self.train_step, self.error])
                print time.time() - tmp
 
                losses.append(error)

                print "Error", error

                if len(losses) > 1 and numpy.abs((losses[-2] - losses[-1]) / losses[-2]) < stopping_threshold:
                    #print stopping_threshold
                    #plt.plot(losses)
                    #plt.show()
                    break

            if i == max_steps - 1:
                print "Warning: Fit exited at max_steps limit, may have not converged"

                #if i % 250 == 0 and i != 0:
                    #plt.plot(losses)
                    #plt.show()
                    #plt.imshow(self.V.eval()[:, :, 0, 0], cmap = plt.cm.gray, interpolation = 'NONE')
                    #plt.colorbar()
                    #plt.show()
                    #import pdb
                    #pdb.set_trace()

            output = sess.run(self.V)[:, :, 0, 0] + sess.run(self.b)[0]

            self.tmpVFile = tempfile.NamedTemporaryFile()

            self.saver.save(sess, self.tmpVFile.name)

        self.update2 = self.x.assign_sub(self.dt * (self.wx2 + self.wy2) * (self.x3 + self.fftV))

        return output 

    def run(self, im, stopping_threshold = 1e-4, max_steps = 50, mean_shift = 0.0):
        """Run the modified Cahn-Hilliard PDE.

        You must first run CahnHilliardWithV.fit first

        Arguments:
        im -- image of size 'shape' given in constructor. The image values will be rescaled to [-1.0 + mean_shift, 1.0 + mean_shift] (required)

        stopping_threshold -- If the average per-pixel percent change (on a [0, 1] scale) in a time step is less than stopping_threshold, the simulation will stop. I.E.,

        xnew = xold + dx
        If (dx / xold) < stopping_threshold, the code will stop.

        max_steps -- If the stopping threshold isn't hit by this number of timesteps, quit

        mean_shift -- Amount by which to shift the input image up or down. Because the Cahn-Hilliard equation is conservative, the mean of the initial conditions will be conserved. What this effects with regards to segmentations is sometimes you find out your image is a bit starved for material to segment out all the precipitates you want. Making this a larger number can help. (Practically it should be [-1.0, 1.0])

        Returns:
        array

        array -- array is an array of size shape that holds the output of the Cahn-Hilliard simulations. It should be easy to actually segment with an Otsu threshold
        """

        import matplotlib.pyplot as plt

        im = util.kmeans_rescale(im) + mean_shift

        if not self.is_fit:
            print "Model is not fit yet. Please run CahnHilliardWithV.fit first"
            return

        with tf.Session() as sess:
            losses = []
            sess.run(tf.initialize_all_variables())
            sess.run(self.initialize, feed_dict = { self.xinp : im })
            self.saver.restore(sess, self.tmpVFile.name)

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

            output = sess.run(self.ix)

        return output
