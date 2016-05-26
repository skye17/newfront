import theano
from RBM import RBM


class CRBM(RBM):
    """An adaptation of RBM to continuous valued-inputs.
    Requires normalized inputs
    Ref: http://www.cs.toronto.edu/~rsalakhu/papers/annrev.pdf (Section 2.2)
    """
    def __init__(self, input=None, n_visible=784, n_hidden=500,
                 W=None, hbias=None, vbias=None, numpy_rng=None,
                 theano_rng=None):
        super(CRBM, self).__init__(input, n_visible, n_hidden,
                                   W, hbias, vbias, numpy_rng, theano_rng)

    def sample_v_given_h(self, h0_sample):
        """ This function infers state of visible units given hidden units """
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.normal(size=v1_mean.shape, avg=v1_mean,
                                           dtype=theano.config.floatX)

        return [pre_sigmoid_v1, v1_mean, v1_sample]
