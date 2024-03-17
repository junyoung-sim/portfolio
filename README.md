# Deep Deterministic Policy Gradient for Simulated Portfolio Optimization with Geometric Brownian Motion

## Objective

Given N independent particles that follow a random walk (i.e, asset price series), train a DDPG agent that maximizes the logarithmic sum of the position of all N particles (i.e., portfolio value).

## Simulating Environment via Geometric Brownian Motion

Suppose a particle's position, $S(t)$, can be modeled as a stochastic process defined by

$$S(t)=S_{0}e^{X(t)}$$

where $X(t)=\sigma B(t) + \mu t$ is Brownian Motion with drift having a lognormal distribution (i.e., Geometric Brownian Motion). By computing the moment generating function for Brownian Motion with drift, the expected position at any given time is simplified as shown below.

$$E(S(t))=S_{0}e^{(\mu+\frac{\sigma^2}{2})t}$$

Given $S_{0}$, $\mu$, $\sigma$, and K, the path of N independent particles following Geometric Brownian Motion can be simulated over K time steps that follow $S_{0}$.

## Deep Deterministic Policy Gradient for Portfolio Optimization



## Parameters

All hyperparameters can be found in ./lib/param.hpp.

## Results

Environment

![alt text](https://github.com/junyoung-sim/portfolio/blob/main/res/path.png)

Train

![alt text](https://github.com/junyoung-sim/portfolio/blob/main/res/log.png)

Test

![alt text](https://github.com/junyoung-sim/portfolio/blob/main/res/test.png)