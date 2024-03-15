#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

path = pd.read_csv("./res/path")
plt.figure()
for key in path:
    plt.plot(path[key], label=key)
plt.savefig("./res/path.png")

log = pd.read_csv("./res/log")
log = log.dropna().reset_index()
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(log["mr"], label="mr")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(log["q"], label="q")
plt.legend()
plt.savefig("./res/log.png")

test = pd.read_csv("./res/test")["test"]
plt.figure()
plt.plot(test, label="model")
plt.legend()
plt.savefig("./res/test.png")