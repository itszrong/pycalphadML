from tensorflow import keras
import tensorflow as tf
from tensorflow_addons.layers import PolynomialCrossing
import pycalphad
from pycalphad import Database, calculate, variables as v

class Rescaler(keras.layers.Layer):
    def __init__(self, temp_scale=1.0):
        super().__init__()
        self.temp_scale = float(temp_scale)
    def build(self, input_shape):
        self.scale_factor = tf.constant([1./self.temp_scale] + ([1.0] * (input_shape[-1]-1)))
    def call(self, inputs):
        scaled_inputs = inputs * self.scale_factor
        return scaled_inputs

class CalphadPhaseModel(keras.models.Model):
    def __init__(self, sublattice_dof, sublattice_site_ratios, temp_scale=1e4, energy_scale=-1e5, name='phase', **kwargs):
        super().__init__(name=name, **kwargs)
        assert len(sublattice_dof) == len(sublattice_site_ratios)
        self.num_statevars = 1
        self.site_ratios = sublattice_site_ratios
        self.sublattice_dof = sublattice_dof
        self.rescaler = Rescaler(temp_scale=temp_scale)
        self.energy_scale = float(energy_scale)
        self.crossnet_1 = PolynomialCrossing(projection_dim=None, use_bias=False)
        self.crossnet_2 = PolynomialCrossing(projection_dim=None, use_bias=False)
        self.crossnet_3 = PolynomialCrossing(projection_dim=None, use_bias=False)
        self.crossnet_4 = PolynomialCrossing(projection_dim=None, use_bias=False)
            
    def call(self, inputs):
        x0 = self.rescaler(inputs)
        output = self.crossnet_1((x0, x0))
        output = self.crossnet_2((x0, output))
        output = self.crossnet_3((x0, output))
        output = self.crossnet_4((x0, output))
        output = self.crossnet_5((x0, output))
        output = self.crossnet_6((x0, output))
        output = self.crossnet_7((x0, output))
        output = self.crossnet_8((x0, output))
        output = self.crossnet_9((x0, output))
        output = self.energy_scale * tf.math.reduce_sum(output, axis=-1)
        dof_idx = int(self.num_statevars)
        ideal_mixing = tf.constant(0.0)
        for num_constituents, ratio in zip(self.sublattice_dof, self.site_ratios):
            ideal_mixing = ideal_mixing + \
                ratio * tf.math.reduce_sum(inputs[..., dof_idx:dof_idx+num_constituents] * tf.math.log(inputs[..., dof_idx:dof_idx+num_constituents]), axis=-1)
            dof_idx += num_constituents
        ideal_mixing = 8.3145 * inputs[..., 0] * ideal_mixing
        return output + ideal_mixing

def fit_surrogate(dbf, comps, phase_name, temp_range, **kwargs):
    mod = pycalphad.Model(dbf, comps, phase_name)
    res = calculate(dbf, comps, phase_name, T=temp_range, P=1e5, N=1, model=mod, pdens=10)
    # Get resulting data into tabular form
    filtered = res.drop_vars('component').to_dataframe()[['Y', 'GM']] \
                    .unstack('internal_dof')\
                    .droplevel(level='component')\
                    .reset_index().drop(columns=['N', 'P', 'points'])
    # Not sure how to fix the duplicate energy column yet
    site_fractions = res.Y.values
    filtered = filtered.values[:, :1+site_fractions.shape[-1]+1]
    x_orig = filtered[:, :-1]
    y_orig = filtered[:, -1]
    # Create Keras Model
    sublattice_dof = [len(t) for t in mod.constituents]
    temp_scale = x_orig[:, 0].max()
    energy_scale = y_orig.std()
    ml_model = CalphadPhaseModel(sublattice_dof, mod.site_ratios, name=phase_name,
                                 temp_scale=temp_scale, energy_scale=energy_scale)
    ml_model.compile(optimizer='adam', loss='mae')
    history = ml_model.fit(x=x_orig, y=y_orig, epochs=200, verbose=0, callbacks=keras.callbacks.TerminateOnNaN(), **kwargs)
    return ml_model, history