import math
import colorsys
from PIL import Image

"""
The plasma effect is a computer-based visual effect animated in real-time. 
It uses cycles of changing colours warped in various ways to give an illusion 
of liquid, organic movement.
"""

def plasma (w, h):
    """
    Plasma effect for width and height w and h, respcetively.
    """
    out = Image.new("RGB", (w, h))
    pix = out.load()
    for x in range (w):
        for y in range(h):
            hue = 4.0 + math.sin(x / 19.0) + math.sin(y / 9.0) \
		      + math.sin((x + y) / 25.0) + math.sin(math.sqrt(x**2.0 + y**2.0) / 8.0)
            hsv = colorsys.hsv_to_rgb(hue/8.0, 1, 1)
	    pix[x, y] = tuple([int(round(c * 255.0)) for c in hsv])
    return out
