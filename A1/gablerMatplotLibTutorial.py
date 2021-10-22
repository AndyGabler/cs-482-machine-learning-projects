# -*- coding: utf-8 -*-
"""
MatplotLib tutorial file.

@author: Andy Gabler
"""

import matplotlib.pyplot as plt
import numpy as np

# CODE FOR EXERCISE 1
fig1 = plt.figure()
fig1, ax = plt.subplots()
ax.plot(np.array([1, 2, 4, 5, 6, 10]), np.array([3, 8, 1, 1, 6, 9]))

# CODE FOR EXERCISE 2
x0 = np.linspace(0, 50, 2)
fig2, ax0 = plt.subplots()
ax0.plot(x0, x0 + 20, label='Cost')
ax0.plot(x0, x0 * 2, label='Revenue')
ax0.set_xlabel("Items Sold")
ax0.set_ylabel("Dollars ($)")
ax0.set_title("Cost-Revenue Projection")
ax0.legend()