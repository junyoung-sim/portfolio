#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

plt.figure(figsize=(40,20))

path = pd.read_csv("./res/path")
plt.subplot(2, 2, 1)
plt.title("Environment")
for key in path:
    plt.plot(path[key], label=key)
plt.legend()

log = pd.read_csv("./res/log")
log = log.dropna().reset_index()
plt.subplot(2, 2, 2)
plt.title("Reward")
plt.plot(log["mr"])

test = pd.read_csv("./res/test")["test"]
plt.subplot(2, 2, 3)
plt.title("Test")
plt.plot(test)

action = pd.read_csv("./res/action")
plt.subplot(2, 2, 4)
plt.title("Action")
for key in action:
    plt.plot(action[key], label=key)
plt.legend()

plt.savefig("./res/result.png")