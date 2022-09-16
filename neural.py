from turtle import shape
from tensorflow import keras
import tensorflow as tf
from tensorflow_addons.layers import PolynomialCrossing
from keras.layers import Dense
from keras.layers import Conv1D
import pycalphad
from pycalphad import Database, calculate, variables as v
from sklearn.model_selection import train_test_split
import numpy as np

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
        self.shape = shape
        self.crossnet_1 = PolynomialCrossing(projection_dim=None, use_bias=False)
        self.crossnet_2 = PolynomialCrossing(projection_dim=None, use_bias=False)
        self.crossnet_3 = PolynomialCrossing(projection_dim=None, use_bias=False)
        self.crossnet_4 = PolynomialCrossing(projection_dim=None, use_bias=False)
        self.crossnet_5 = PolynomialCrossing(projection_dim=None, use_bias=False)
        self.crossnet_6 = PolynomialCrossing(projection_dim=None, use_bias=False)
        self.crossnet_7 = PolynomialCrossing(projection_dim=None, use_bias=False)
        self.crossnet_8 = PolynomialCrossing(projection_dim=None, use_bias=False)
        # self.crossnet_9 = PolynomialCrossing(projection_dim=None, use_bias=False)

        # self.crossnet_9 = PolynomialCrossing(projection_dim=None, use_bias=False, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.1, l2=0.01))

        # self.linear = Linear()
        # self.dense1 = Dense(256, activation=tf.nn.sigmoid)
        # self.dense2 = Dense(7, activation=tf.nn.sigmoid)

        # self.dense1 = Dense(units=1)
        # self.dense2 = Dense(units=1)
        # self.dense3 = Dense(units=1)
        # self.dense4 = Dense(units=1)
        # self.dense5 = Dense(units=1)
        # self.dense6 = Dense(units=1)
        # self.dense7 = Dense(units=1)
        # self.dense8 = Dense(units=1)
        # self.dense9 = Dense(units=1)
        # self.dense10 = Dense(units=1)
        
        # , kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.1, l2=0.01)


    def call(self, inputs):

        # X = inputs
        # T, XMg, XSi, XZn  = X[:,0], X[:,1], X[:,2], X[:,3]
        # fit = self.dense1(tf.expand_dims(XMg*XZn*(T*tf.math.log(T)), 1))
        # fit += self.dense2(tf.expand_dims(XMg*XZn*(XMg-XZn)*(T*tf.math.log(T)), 1))
        # fit += self.dense3(tf.expand_dims(XMg*XZn*(XMg-XZn)**2*(T*tf.math.log(T)),1))
        # fit += self.dense4(tf.expand_dims(XMg*XSi*(T*tf.math.log(T)),1))
        # fit += self.dense5(tf.expand_dims(XMg*XSi*(XMg-XSi)*(T*tf.math.log(T)),1))
        # fit += self.dense6(tf.expand_dims(XMg*XSi*(XMg-XSi)**2*(T*tf.math.log(T)),1))
        # fit += self.dense7(tf.expand_dims(XSi*XZn*(T*tf.math.log(T)),1))
        # fit += self.dense8(tf.expand_dims(XSi*XMg*(XSi-XMg)*(T*tf.math.log(T)),1))
        # fit += self.dense9(tf.expand_dims(XSi*XMg*(XSi-XMg)**2*(T*tf.math.log(T)),1))
        # fit += self.dense10(tf.expand_dims(XMg*XSi*XMg*(T*tf.math.log(T)),1))

        x0 = self.rescaler(inputs)
        output = self.crossnet_1((x0, x0))
        output = self.crossnet_2((x0, output))
        output = self.crossnet_3((x0, output))
        output = self.crossnet_4((x0, output))
        # output = self.crossnet_5((x0, output))
        # output = self.crossnet_6((x0, output))
        # output = self.crossnet_7((x0, output))
        # output = self.crossnet_8((x0, output))
        # term = self.dense3((tf.ones(32,1), tf.transpose((inputs[:,0]*tf.math.log(inputs[:,0])[np.newaxis]))))
        # term = self.dense3(tf.math.divide(tf.transpose((inputs[:,0]*tf.math.log(inputs[:,0]))[np.newaxis]), 1000))
        # output = self.dense1(inputs)
        # output = self.dense2(output)
        # # output = tf.math.reduce_sum(self.dense2(output), axis=-1)
        # output = self.dense3(output)
        # output = self.energy_scale * (tf.math.reduce_sum(output, axis=-1)) + tf.math.reduce_sum(fit, axis=-1)
        output = self.energy_scale * (tf.math.reduce_sum(output, axis=-1))
        dof_idx = int(self.num_statevars)
        ideal_mixing = tf.constant(0.0)
        for num_constituents, ratio in zip(self.sublattice_dof, self.site_ratios):
            ideal_mixing = ideal_mixing + \
                ratio * tf.math.reduce_sum(inputs[..., dof_idx:dof_idx+num_constituents] * tf.math.log(inputs[..., dof_idx:dof_idx+num_constituents]), axis=-1)
            dof_idx += num_constituents
        ideal_mixing = 8.3145 * inputs[..., 0] * ideal_mixing
        return output

# def fit_surrogate(dbf, comps, phase_name, temp_range, **kwargs):
#     mod = pycalphad.Model(dbf, comps, phase_name)
#     res = calculate(dbf, comps, phase_name, T=temp_range, P=1e5, N=1, model=mod, pdens=10)
#     # Get resulting data into tabular form
#     filtered = res.drop_vars('component').to_dataframe()[['Y', 'GM']] \
#                     .unstack('internal_dof')\
#                     .droplevel(level='component')\
#                     .reset_index().drop(columns=['N', 'P', 'points'])
#     # Not sure how to fix the duplicate energy column yet
#     site_fractions = res.Y.values
#     filtered = filtered.values[:, :1+site_fractions.shape[-1]+1]
#     x_orig = filtered[:, :-1]
#     y_orig = filtered[:, -1]
#     # Create Keras Model
#     sublattice_dof = [len(t) for t in mod.constituents]
#     temp_scale = x_orig[:, 0].max()
#     print('temp_scale is', temp_scale)
#     energy_scale = y_orig.std()

#     ml_model = CalphadPhaseModel(sublattice_dof, mod.site_ratios, name=phase_name,
#                                  temp_scale=temp_scale, energy_scale=energy_scale)
#     ml_model.compile(optimizer='adam', loss='mae')
#     history = ml_model.fit(x=x_orig, y=y_orig, epochs=20, verbose=1, callbacks=keras.callbacks.TerminateOnNaN(), **kwargs)
#     return ml_model, history


def grad_ML(inputs, model):
    tfinputs = tf.convert_to_tensor(inputs)

    with tf.GradientTape() as t:
      t.watch(tfinputs)
      output = model(tfinputs)

    dz_dx = t.gradient(output, tfinputs)
    print(dz_dx.numpy())
    gradient_T = dz_dx.numpy()[:,0:1]
    print(gradient_T.shape)
    return gradient_T

def custom_loss_function(y_true, y_pred):
    # error = tf.square(y_true-y_pred)

    # mse_loss = keras.losses.mean_squared_error(y_true, y_pred)
    # derivative_loss = keras.losses.mean_squared_error(input_tensor, -grad(input_tensor, grad(input_tensor, output_tensor))[0])

    RMSE =  tf.sqrt(tf.divide(tf.reduce_sum(tf.pow(tf.subtract(y_true, y_pred),2.0)),tf.cast(tf.size(y_true), tf.float32)))       
    print('Shapes', y_true.shape, y_pred.shape)
    # return mse_loss + derivative_loss
    return RMSE

    # return tf.reduce_mean(error, axis=-1)

def fit_surrogate(dbf, comps, phase_name, temp_range, test_set_size, epochs_per_run, **kwargs):
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
    print('temp_scale is', temp_scale)
    energy_scale = y_orig.std()

    x_train, x_test, y_train, y_test = train_test_split(x_orig, y_orig, test_size=test_set_size, random_state = 42)
    # print(x_train.shape, y_train.shape)

    ml_model = CalphadPhaseModel(sublattice_dof, mod.site_ratios, name=phase_name,
                                 temp_scale=temp_scale, energy_scale=energy_scale)
    # ml_model.compile(optimizer='adam', loss='mae')
    
    ml_model.compile(optimizer='adam', loss='mae')
    history = ml_model.fit(x=x_train, y=y_train, epochs=epochs_per_run, verbose=1, callbacks=keras.callbacks.TerminateOnNaN(), **kwargs)
    return ml_model, history, x_train, y_train, x_test, y_test