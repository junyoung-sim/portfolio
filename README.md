# Deep Deterministic Policy Gradient for Simulated Portfolio Optimization with Geometric Brownian Motion

## Objective

Given N independent particles that follow a random walk (i.e, asset price series), train a DDPG agent that maximizes the logarithmic sum of the position of all N particles (i.e., portfolio value).

## Simulating Environment via Geometric Brownian Motion

Suppose a particle's position, $P(t)$, can be modeled as a stochastic process defined by

$$P(t)=P_{0}e^{X(t)}$$

where $X(t)={\sigma}B(t)+{\mu} t$ is Brownian Motion with drift having a lognormal distribution (i.e., Geometric Brownian Motion). By computing the moment generating function for Brownian Motion with drift, the expected position at any given time is simplified as shown below.

$$E(P(t))=P_{0}e^{(\mu+\frac{\sigma^2}{2})t}$$

Given $P_{0}$, $\mu$, $\sigma$, and K, the path of N independent particles following Geometric Brownian Motion can be simulated over K time steps that follow $P_{0}$.

## Deep Deterministic Policy Gradient for Portfolio Optimization

Given some state $s_t$, the actor network with parameters $\phi$ yields the action space $\mu(s_{t}|\phi)=a_{t}$ that is responded by some reward function $r_{t}$. As the actor network must maximize reward, the following objective function is maximized via gradient ascent with respect to $\phi$.

$$J=Q(s_{t},a_{t})$$

$Q(s_{t},a_{t})$ is the expected reward (i.e., q-value), predicted by the critic network with parameters $\theta$, given the state-action space $s_{t}$ and $a_{t}$. As $J$ is maximized, the critic network's q-value approximation should also converge to the Bellman Equation shown below via gradient descent with respect to $\theta$. Note that $Q'$ is obtained from a delayed copy (i.e., target) of the critic network. Likewise, $\mu'$ is a delayed copy of the actor network. Soft update with $\tau$ must be used to synchronize the real-time and delayed versions by copying a small percentage of the real-time parameters to their targets after every batch update.

$$Q^{*}(s_{t},a_{t})=r_{t}+{\gamma}Q'(s_{t+1},a_{t+1}=\mu'(s_{t+1}))$$

Since the critic network directly maps the state-action space to reward, the action gradient $\frac{dQ}{da}$ must be computed for each action while updating the critic network such that ${\nabla}J={\nabla}_{a_{t}}Q(s_{t},a_{t};\theta){\nabla}_{\phi}\mu(s_{t})$ can be computed to maximize $J$. Exploration can be done by either adding OU noise or uncorrelated Gaussian noise to the parameters.

Let the state space be $s_{t}=(P_{1}(t), P_{2}(t), ..., P_{n-1}(t), P_{n}(t))$ where $P_{i}(t)$ is the most recent position of each independent particle. Given $s_t$, the actor yields the action space $a_{t}$ where each value in $a_{t}$ is the percentage allocation corresponding to each independent particle's position. The reward is calculated as $r_{t}=\log(s_{t+1}{\cdot}a_{t})$.

## Parameters

All hyperparameters can be found in ./lib/param.hpp.

## Results

**Environment**
![alt text](https://github.com/junyoung-sim/portfolio/blob/main/res/path.png)

**Train**
![alt text](https://github.com/junyoung-sim/portfolio/blob/main/res/log.png)

**Test**
![alt text](https://github.com/junyoung-sim/portfolio/blob/main/res/test.png)