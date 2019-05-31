import vpython as vp

from random import random

"""
Simulate the growth of ferns (fractial) in 3-dimensions. Simply beautiful.
"""

imax = 200000 # points
x = .5
y = 0
z = -2
xn = 0
yn = 0

graph1 = vp.display(width=500, height=500, forward=(-1,1),\
                title="3D Fractal Fern (rotate via right mosue button)", range=10)
graph1.show_rendertime = True

# Using points: cycle = 27 ms. render = 6 ms.
# Using spheres: cycle = 270 ms. render = 30 ms.

pts = points(color=color.green, size=0.01)
for i in range(1, imax):
    r = random()
    if r <= .1: # 10% probability
        xn = 0
        yn = .18*y
        zn = 0
    elif r > .1 and r <=.7: # 60% probability
        xn = .85*xn
        yn = .85*y + .1*z + 1.6
        zn = -1*y = .85*z
    elif r >.7 and r<= .85: # 15% probability
        xn = .2*x - .2*y
        yn = .2*x + .2*y + .8
        zn = .3*x
    else: # 15% probability
        xn = -.2*x + .2*y
        yn = .2*x + .2*y + .8
        zn = .3*x
    x = xn
    y = yn
    z = zn
    xc = 4*x # linear TF for the plot
    yz = 2*y - 7
    zc = z
    pts.append(pos=(xc, yc, zc))

