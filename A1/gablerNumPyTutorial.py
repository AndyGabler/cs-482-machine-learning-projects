# -*- coding: utf-8 -*-
"""
NumPy tutorial file.

@author: Andy Gabler
"""

import numpy as np

# CODE FOR EXERCISE 1
e1 = np.array([1, 2, 3, 4])
print("\nExercise 1\n{}".format(e1))

# CODE FOR EXERCISE 2
b = np.zeros((2, 7))
print("\nExercise 2\n{}".format(b))

# CODE FOR EXERCISE 3
c = np.arange(1, 23, 2.5) # Note, inclusive however, 23.5 is next increment so stops at 21.0
print("\nExercise 3\n{}".format(c))

# CODE FOR EXERCISE 4
d = np.arange(1, 41, 1).reshape(5, 8)
print("\nExercise 4\n{}".format(d[4, 7]))

# CODE FOR EXERCISE 5
arr = np.arange(1, 19, 1).reshape(3, 3, 2)
# No print asked for in instructions. These are present for debugging purposes
#print("Array arr Dimensions: ", arr.ndim)
#print("Array arr Item Size: ", arr.itemsize)
#print("Array arr Size: ", arr.shape)
#print("Array arr Size: ", arr.size)
#print("Array arr Data Type: ", arr.dtype.name)

# CODE FOR EXERCISE 6
# remember @ is matrix product
m = np.random.random((4, 4))
p = np.random.random((4, 4))
k = m.sum()
print("\nExercise 6\n{}".format(np.sin(m) @ np.cos(p) + k))

# CODE FOR EXERCISE 7
print("\nExercise 7")
f1 = np.random.random((1, 10))
f1Sorted = np.sort(f1)
print("Smallest: {}".format(f1Sorted[0, 0]))
print("Largest: {}".format(f1Sorted[0, 9]))
f2 = f1.reshape(2, 5)
print("Concatenation F2+F2:\n{}".format(np.concatenate((f2, f2), axis=0)))

# CODE FOR EXERCISE 8
import math
print("\nExercise 8")
A1 = np.arange(10)
print("Initial Array (A1):\n{}".format(A1))
# Even index print. (reshape so even indexes are in first column, indexes starting at 0 of course)
print("Even elements of A1:\n{}".format(A1.reshape((5, 2))[:, 0]))
print("First four elements of A1:\n{}".format(A1[:4]))
print("Last 3 elements of A1:\n{}".format(A1[-3:]))
print("Elements 4-8 of A1:\n{}".format(A1[4:8]))
np.random.seed(42)
M1 = 100 * np.random.rand(9, 7).round(2)
print("Initial Array (M1):\n{}".format(M1))
print("First and Last Row of M1:\n{}".format(M1[(-1, 0), :]))
print("Elements of M1 <10:\n{}".format(M1[M1<10]))
print("Even Rows of M1:\n{}".format(M1[np.arange(0, M1.shape[0], 2)]))
print("Odd Columns of M1:\n{}".format(M1[:, np.arange(1, M1.shape[1], 2)]))
print("Even Rows Odd Columns of M1:\n{}".format(M1[np.arange(0, M1.shape[0], 2)][:, np.arange(1, M1.shape[1], 2)]))
print("Even Indice Pairings of M1:\n{}\n".format(M1.flat[np.arange(0, M1.shape[0] * M1.shape[1], 2)]))
