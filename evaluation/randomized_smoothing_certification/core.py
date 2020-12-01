'''
Randomized Smoothing: Hard-RS and Soft-RS
Soft-RS uses empirical Bernstein bound
MACER: Attack-free and Scalable Robust Training via Maximizing Certified Radius
ICLR 2020 Submission
References:
[1] J. Cohen, E. Rosenfeld and Z. Kolter. 
Certified Adversarial Robustness via Randomized Smoothing. In ICML, 2019.
Acknowledgements:
[1] https://github.com/locuslab/smoothing/blob/master/code/core.py
'''

from math import ceil
from scipy.stats import norm, binom_test
import numpy as np
from statsmodels.stats.proportion import proportion_confint

class Smooth(object):
    '''
    Smoothed classifier
    mode can be hard, soft or both
    beta is the inverse of softmax temperature
    '''

    # to abstain, Smooth returns this int
    ABSTAIN = -1
    
    def __init__(self, sess, x_eval_tf, cls_test, num_classes, sigma, batch_size):
        self.sess = sess
        self.x_eval_tf = x_eval_tf
        self.cls_test = cls_test
        self.num_classes = num_classes
        self.sigma = sigma
        self.batch_size = batch_size
     
    def certify(self, X_test_image, n0, n, alpha, sigma_val):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.
        
        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(X_test_image, n0, sigma_val)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(X_test_image, n, sigma_val)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = sigma_val[0] * norm.ppf(pABar)
            return cAHat, radius
        
    
    def predict(self, X_test_image, n, alpha, sigma_val):
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).
        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.
        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """

        counts = self._sample_noise(X_test_image, n, sigma_val)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]
    
    def _sample_noise(self, X_test_image, num, sigma_val):
        """ Sample the base classifier's prediction under noisy corruptions of the input x.
        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        counts = np.zeros(self.num_classes, dtype=int)
        for _ in range(ceil(num / self.batch_size)):
            this_batch_size = min(self.batch_size, num)
            num -= this_batch_size
            
            predictions = self.sess.run(self.cls_test, feed_dict = {self.x_eval_tf: X_test_image, self.sigma:sigma_val}).argmax(1)
            counts += self._count_arr(predictions, self.num_classes)
        return counts
    
    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts    
    
    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.
        This function uses the Clopper-Pearson method.
        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]