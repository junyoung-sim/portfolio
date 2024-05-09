# Deep Deterministic Policy Gradient and Geometric Brownian Motion for Simulated Portfolio Optimization

## Objective

Given N independently moving price series following geometric Brownian Motion with equivalent parameters, train a DDPG agent that maximizes the logarithmic portfolio value.

This is a 4-year long culmination of multiple quantitative trading algorithm projects listed below:

1) https://github.com/junyoung-sim/sltm

2) https://github.com/junyoung-sim/quant

3) https://github.com/junyoung-sim/gbm-drl-quant

4) https://github.com/junyoung-sim/ddpg-quant

## Simulating Environment via Geometric Brownian Motion

Suppose an asset's price, $P(t)$, can be modeled as a stochastic process defined by

$$P(t)=P_{0}e^{X(t)}$$

where $X(t)={\sigma}B(t)+{\mu} t$ is Brownian Motion with drift having a lognormal distribution (i.e., Geometric Brownian Motion). By computing the moment generating function for Brownian Motion with drift, the expected price at any given time is simplified as shown below.

$$E(P(t))=P_{0}e^{(\mu+\frac{\sigma^2}{2})t}$$

Given $P_{0}$, $\mu$, $\sigma$, and K, the path of N independent price series following Geometric Brownian Motion can be simulated over K time steps that follow $P_{0}$.

## Deep Deterministic Policy Gradient for Portfolio Optimization

Given some state $s_t$, the actor network with parameters $\phi$ yields the action space $\mu(s_{t}|\phi)=a_{t}$ that is responded by some reward function $r_{t}$. As the actor network must maximize reward, the following objective function is maximized via gradient ascent with respect to $\phi$.

$$J=Q(s_{t},a_{t})$$

$Q(s_{t},a_{t})$ is the expected reward (i.e., q-value), predicted by the critic network with parameters $\theta$, given the state-action space $s_{t}$ and $a_{t}$. As $J$ is maximized, the critic network's q-value approximation should also converge to the Bellman Equation shown below via gradient descent with respect to $\theta$. Note that $Q'$ is obtained from a delayed copy (i.e., target) of the critic network. Likewise, $\mu'$ is a delayed copy of the actor network. Soft update with $\tau$ must be used to synchronize the real-time and delayed versions by copying a small percentage of the real-time parameters to their targets after every batch update from replay memory.

$$Q^{*}(s_{t},a_{t})=r_{t}+{\gamma}Q'(s_{t+1},a_{t+1}=\mu'(s_{t+1}))$$

Since the critic network directly maps the state-action space to reward, the action gradient $\frac{dQ}{da}$ must be computed for each action while updating the critic network such that $\nabla{J}=\nabla_{a_{t}}Q(s_{t},a_{t};\theta)\nabla_{\phi}\mu(s_{t})$ can be computed to maximize $J$. Exploration can be done by either adding OU noise or uncorrelated Gaussian noise to the parameters.

Let the state space be $s_{t}=<\delta P_{1}(t), \delta P_{2}(t), ..., \delta P_{n-1}(t)>$ where $\delta P_{i}(t)$ is the percentage change of each asset's price series over some time-step. Given $s_t$, the actor yields the action space $a_{t}$ where each value in $a_{t}$ is the weight allocated to each asset. The reward is calculated as $r_{t}=\log(P_{t+1}{\cdot}a_{t})$.

## Parameters

All hyperparameters can be found in ./lib/param.hpp.

## Results

![alt text](https://github.com/junyoung-sim/portfolio/blob/main/res/trial1/result.png)

![alt text](https://github.com/junyoung-sim/portfolio/blob/main/res/trial2/result.png)

![alt text](https://github.com/junyoung-sim/portfolio/blob/main/res/trial3/result.png)

The results above show that the model tends to maximize holdings of assets with the best momentum. Furthermore, the model demonstrates arbitrage behavior as its final portfolio value closely reflects the average value of all N assets.

Note that these results above are merely experimental as the model's convergence and behavior may vary by the random seed assigned for simulating an environment. Further work and more complex optimization objectives would be needed for practical applications in real market environments.

## References

https://www.columbia.edu/~ks20/FE-Notes/4700-07-Notes-GBM.pdf

https://arxiv.org/abs/1509.02971

https://spinningup.openai.com/en/latest/algorithms/ddpg.html