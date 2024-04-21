#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

plt.figure(figsize=(25, 5))

path = pd.read_csv("./res/path")
plt.subplot(1, 3, 1)
plt.title("Environment")
for key in path:
    plt.plot(path[key])

log = pd.read_csv("./res/log")
log = log.dropna().reset_index()
plt.subplot(1, 3, 2)
plt.title("Reward")
plt.plot(log["mr"])

test = pd.read_csv("./res/test")["test"]
plt.subplot(1, 3, 3)
plt.title("Test")
plt.plot(test)

plt.savefig("./res/result.png")