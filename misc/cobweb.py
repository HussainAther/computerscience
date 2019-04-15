import numpy as np
import pylab as plt

"""
Math follows nature
"""

a=0.66667
k=0.0025
x=0.2
 
def f(x):
    """
    Vary our function periodically.
    """
    return (x+a+ k*np.sin(2.*np.pi*x))%1.0

xlo=0.0
xhi=1.0
xt=np.linspace(xlo,xhi,10)
fxt=f(xt)

ax=plt.subplot(111,aspect='equal')

plt.plot(xt,fxt,'ko',linewidth=4)
plt.plot(xt,xt,'k-',linewidth=1)

def vertical(x):
    plt.plot( [x,x],[ x ,f(x)],'k--',linewidth=2)

def across(x):
    plt.plot( [x,f(x)],[f(x),f(x)],'k--',linewidth=2)

plt.plot( [x,x],[xlo,f(x)],'k--',linewidth=2)

for i in range(200):
    if i>0: vertical(x)
    across(x)
    x=f(x)

plt.xlabel(r"$\phi_{\rm n}$", fontsize=24)
plt.ylabel(r"$\phi_{\rm n+1}$", fontsize=24)

plt.savefig("miltonohira_11_14.png", dpi=600)
plt.savefig("miltonohira_11_14.eps", dpi=600)

plt.show()
