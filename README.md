# Simulate Binomial distribution from Bernoulli trials

The module generates independents Bernoulli experiments with specified success rate (p). 
The n independent Bernoulli experiments are combined to simulate a Binomial (n, p)
Fimally, a histogram is produced using k_samples from the simulated binomials to 
visually assess the approximation of the binomial. 
Also, a check is implemented to show the proportion of the k_samples binomials that have 
their mean and stderr within a relative tolerance range.

