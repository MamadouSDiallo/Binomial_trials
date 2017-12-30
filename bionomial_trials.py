'''  
Independent bernoulli trials as a way to obtain a Binomial experiment.

Author: Mamadou S Diallo 
'''

import numpy as np
from numpy import random as rand

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


class NotProbabilityError(Exception):
    pass


class BernoulliTrial():
    ''' representation of a Bernoulli trial '''

    def __init__(self, p):
        self.p = p
        try:
            if (p < 0 and p > 1):
                raise NotProbabilityError(
                    p, 'is not a valid probability value, '
                    + 'please enter a number between 0 and 1'
                )
            elif p == 0:
                print('The Bernoulli trial will return 0=FAIL with certaintly')
            elif p == 1:
                print('The Bernoulli trial will return 1=SUCCESS with certaintly')
        except NotProbabilityError as notproba:
            print(notproba)

    def __mean_bernoulli__(self):
        ''' mean of Bernoulli distribution '''
        mean_bernoulli = self.p
        return(mean_bernoulli)

    def __stderr_bernoulli__(self):
        ''' standard error of Bernoulli distribution '''
        stderr_bernoulli = np.sqrt(self.p * (1 - self.p))
        return(stderr_bernoulli)

    def print_bernoulli(self):
        print('This is a Ber(' + str(self.__mean_bernoulli__()) + ') distribution')


class BinomialTrial(BernoulliTrial):
    ''' representation of a Binomial as a independent Bernoulli trials '''

    def __init__(self, p, n, tolerance, seed):
        super().__init__(p)
        self.n = n
        self.p = p
        self.seed = seed
        self.tolerance = tolerance
        self.mean_binomial = n * super().__mean_bernoulli__()
        self.stderr_binomial = np.sqrt(n) * super().__stderr_bernoulli__()

    def n_Bernoulli_trials(self):
        ''' Binomial construction '''
        np.random.seed(self.seed)
        vect_trials = rand.random(self.n)
        vect_outcomes = (vect_trials <= self.p)
        return(vect_outcomes)

    def sample_mean(self):
        ''' calculates the sample mean and stand error for the Binomial'''
        sample_mean_binomial = sum(self.n_Bernoulli_trials())
        return(sample_mean_binomial)

    def sample_stderr(self):
        ''' calculates the sample mean and stand error for the Binomial'''
        bern_p = self.sample_mean() / self.n
        sample_stderr_binomial = np.sqrt(self.n * bern_p * (1 - bern_p))
        return(sample_stderr_binomial)

    def check_binomial(self):
        ''' Checks if the construction is binomial.
            That is, it checks that two parameters are as expected i.e.
            the theoritical parameters are compared to the ones obtained from the sample
        '''
        ratio_mean = abs(self.mean_binomial -
                         self.sample_mean()) / self.mean_binomial
        ratio_stderr = abs(self.stderr_binomial - 
                           self.sample_stderr()) / self.stderr_binomial
        if (ratio_mean < self.tolerance and ratio_stderr < self.tolerance):
            isBinomial = True
        else:
            isBinomial = False
        return(isBinomial)

    def print_binomial(self):
        print('This is a Bin(' + str(self.n) + ', ' + str(super().__mean_bernoulli__())
              + ') distribution')




''' The code below produce a histogram of the generated binomials and 
    overlay the equivalent binomial distribution to visually assess
    the empirical distribution
'''

# Produce an histogram of k_samples binomial samples with p=0.4 and n=50000
np.random.seed(342431)

p=0.4
n=5000
k_samples = 1000
tolerance = 0.03
k_seed = rand.randint(1, 1e7, k_samples)
k_binomials = np.zeros(k_samples)
k_checks = np.zeros(k_samples, dtype=bool)

#print(BinomialTrial(p, n).sample_mean())

# Print teh parameters of the Bernoulli and Binomial distribution
bernoulli = BernoulliTrial(p)
bernoulli.print_bernoulli()

binomial = BinomialTrial(p, n, tolerance, 23441)
binomial.print_binomial()

#print(k_binomials[range(5)])
for k in range(k_samples):
    #print(k_seed[k])
    binomial = BinomialTrial(p, n, tolerance, k_seed[k])
    k_binomials[k] = binomial.sample_mean()
    k_checks[k] = binomial.check_binomial()

#print(k_binomials[range(15)])
#print(k_checks[range(15)])

# Check the empirical parameters
percentage_pass = 100 * sum(k_checks) / k_samples
print('Percentage of samples that are close enough: '+str(percentage_pass)+'%')

# Histogram
mu = n * p
sigma = np.sqrt(n * p * (1 - p))

num_bins = 20
nn, bins, patches = plt.hist(
    k_binomials, bins=num_bins, normed=1, color='green', alpha=0.6
)

# add a 'best fit' line
y = mlab.normpdf(bins, mu, sigma)

plt.plot(bins, y, 'r--')

plt.xlabel('Number of successes')
plt.ylabel('Density')
plt.title('Histogram of ' + str(k_samples) +
          ' Binomial(' + str(n) + ', ' + str(p) + ') trials \n')

plt.subplots_adjust(left=0.15)
plt.show

''' import pandas as pd
k_binomials = pd.DataFrame(np.zeros(k_samples), columns=['Success rate'])
k_binomials = k_binomials.apply(BinomialTrial(p, n).sample_mean()) '''

