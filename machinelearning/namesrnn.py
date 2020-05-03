import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

"""
Generate names with recurrent neural networks.
"""

start_token = " "

with open("names") as f:
    names = f.read()[:-1].split("\n")
    names = [start_token+name for name in names]
