from random import seed, gauss
import numpy as np
import matplotlib.pyplot as plt

import math

media = 5.0
sigma1 = 0.2
sigma2 = 0.5
sigma3 = 0.8

x = np.linspace(0, 10, 100)

f1 = (1.0/(sigma1*np.sqrt(2.0*math.pi)))*np.exp(-0.5*((x-media)/sigma1)**2.0)
f2 = (1.0/(sigma2*np.sqrt(2.0*math.pi)))*np.exp(-0.5*((x-media)/sigma2)**2.0)
f3 = (1.0/(sigma3*np.sqrt(2.0*math.pi)))*np.exp(-0.5*((x-media)/sigma3)**2.0)

plt.plot(x,f1,label=r'$\sigma$='+str(sigma1))
plt.plot(x,f2,label=r'$\sigma$='+str(sigma2))
plt.plot(x,f3,label=r'$\sigma$='+str(sigma3))
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.tight_layout()
plt.show()

seed(1)

sigma = 1
media = 0

valores = [gauss(media, sigma) for i in range(0, 20000)]

plt.hist(valores, bins=20)
plt.xlabel('X')
plt.ylabel('Contagem')
plt.show()
