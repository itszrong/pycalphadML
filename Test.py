from neural import fit_surrogate
import numpy as np
import pycalphad
from pycalphad import Database, calculate, variables as v

starting_temp = 300
max_temp = 600
step = 5
dbf = Database('Mg_Si_Zn.tdb')
comps = ['MG', 'SI', 'ZN', 'VA']

fine_temps = np.arange(starting_temp, max_temp, step)
print(fine_temps.shape)
pts = np.array([[[0.3, 0.3, 0.4]]])

res = calculate(dbf, comps, 'LIQUID', T=fine_temps, P=1e5, N=1, pdens=10, output='HM', points=pts)
res.HM.shape
