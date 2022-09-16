import numpy as np
import pycalphad
from pycalphad import Database, calculate, variables as v
from pycalphad.core.composition_set import CompositionSet
from pycalphad.core.phase_rec import PhaseRecord
from pycalphad.core.utils import unpack_components, instantiate_models
from pycalphad.codegen.callables import build_phase_records
import matplotlib.pyplot as plt

def get_values(output_type, phase, x_test, dbf, comps):
    species = sorted(unpack_components(dbf, comps), key=str)
    models = {}
    phase_records = {}

    if models.get(phase, None) is None:
        models[phase] = instantiate_models(dbf, species, [phase], parameters=None)[phase]
    if phase_records.get(phase, None) is None:
        phase_records[phase] = build_phase_records(dbf, species, [phase],
                                                    {v.T}, models, output=output_type, build_gradients=True, build_hessians=True)[phase]
    prx = phase_records[phase]
    out = np.zeros(x_test.shape[0])
    # print(out.shape, x_test.shape)
    prx.obj_2d(out, x_test)

    #pts = np.array([[[0.3, 0.3, 0.4]]])
    #res_GM = calculate(dbf, comps, 'LIQUID', T=x_test[:,0], P=1e5, N=1, pdens=10, output='GM', points=x_test[:,1:4])
    # print('test')
    # print(out)
    return out