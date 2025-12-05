import numpy as np
from matplotlib import pyplot as plt

project = 'cell'
seed = 8

fname = f'{project}/thermo/{seed}.txt'
fname_out = f'{project}/plot_{seed}.png'
df = np.loadtxt(fname, delimiter=';', comments='#')

t = df[:, 0]
T = df[:, 1]
pe = df[:, 2]
ke = df[:, 3]
max_ke = df[:, 4]

pe -= pe[0]
ke -= ke[0]

plt.plot(t, ke, color='red')
plt.plot(t, pe, color='blue')
plt.twinx()
plt.plot(t, max_ke, color='black')
plt.savefig(fname_out)
plt.show()