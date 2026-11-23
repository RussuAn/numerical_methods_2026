import math

def f(x):
    return math.sin(x) - 0.3 * x + 0.1

def df(x):
    return math.cos(x) - 0.3

def d2f(x):
    return -math.sin(x)