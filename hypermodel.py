'''HyperModel'''
import random
import tensorflow as tf
from tensorflow import keras
import keras_tuner
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l1, l2
import numpy as np

class TheHyperModel(keras_tuner.HyperModel):
    '''Class for HyperModel which is required by Keras Tuner'''

    def __init__(self, envir, hidden_layers_min_val, hidden_layers_max_val,
                 hidden_units_min_val, hidden_units_max_val, hidden_units_step_val,
                 activation_val, dropout_rate_val, lr_min_val, lr_max_val, lr_sampling_val,
                 optimizer_val, reg_value_val, reg_val, loss_funct_val):

        self.envir = envir
        self.input_dim=len(self.envir.cols)
        self.hidden_layers_min_val = hidden_layers_min_val
        self.hidden_layers_max_val = hidden_layers_max_val
        self.hidden_units_min_val = hidden_units_min_val
        self.hidden_units_max_val = hidden_units_max_val
        self.hidden_units_step_val = hidden_units_step_val
        self.activation_val = activation_val
        self.dropout_rate_val = dropout_rate_val
        self.lr_min_val = lr_min_val
        self.lr_max_val = lr_max_val
        self.lr_sampling_val = lr_sampling_val
        self.optimizer_val = optimizer_val
        self.reg_value_val = reg_value_val
        self.reg_val = reg_val
        self.loss_funct_val = loss_funct_val

    def set_seeds(self, seed=100):
        '''Initialize random process'''
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def create_model(self, hidden_layers, hidden_units, activation, dropout, dropout_rate,
                     optimizer, lr, regularize, reg, loss_funct):
        '''Define Neural Network Model'''

        if not regularize:
            reg = None

        model = Sequential()
        model.add(Dense(units=hidden_units, activation=activation, use_bias=True,
                  kernel_initializer="glorot_uniform", bias_initializer="zeros",
                  kernel_regularizer=None, bias_regularizer=None, activity_regularizer=reg,
                  kernel_constraint=None, bias_constraint=None, input_dim=self.input_dim))

        if dropout:
            model.add(Dropout(dropout_rate, seed=100))

        for _ in range(hidden_layers):
            model.add(Dense(units=hidden_units, activation=activation, activity_regularizer=reg))
            if dropout:
                model.add(Dropout(dropout_rate, seed=100))

        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss=loss_funct, optimizer=optimizer, metrics=['accuracy'])
        return model

    def build(self, hp):
        '''Build Neural Network Model'''

        # Model layers and units/neurons
        hidden_layers = hp.Int('layers', min_value=self.hidden_layers_min_val,
                                         max_value=self.hidden_layers_max_val)
        hidden_units = hp.Int('units', min_value=self.hidden_units_min_val,
                                       max_value=self.hidden_units_max_val,
                                       step=self.hidden_units_step_val)

        # Activation function for dense layers
        activation = hp.Choice('activation', self.activation_val)

        # Dropout yes/no and dropout rate
        dropout = hp.Boolean('dropout')
        dropout_rate = hp.Choice('drop_rate', self.dropout_rate_val)

        # Optimizer type and learning rate; 'adamax' removed due to error message
        lr = hp.Float('lr', min_value=self.lr_min_val,
                            max_value=self.lr_max_val,
                            sampling=self.lr_sampling_val)
        optimizer = self.opt_wrapper(hp.Choice('optimizer', self.optimizer_val), lr)

        # Regularizer type and factor -> equality in weight levels
        regularize = hp.Boolean('regularize')
        reg_value = hp.Choice('reg_value', self.reg_value_val)
        reg = self.reg_wrapper(hp.Choice('type', self.reg_val), reg_value)

        # Loss function
        loss_funct = hp.Choice('loss_funct', self.loss_funct_val)

        # Call main model creation module and transfer model to keras tuner
        self.set_seeds()
        model = self.create_model(hidden_layers=hidden_layers, hidden_units=hidden_units,
                                  activation=activation, dropout=dropout, dropout_rate=dropout_rate,
                                  optimizer=optimizer, lr=lr,
                                  regularize=regularize, reg=reg,
                                  loss_funct=loss_funct)

        return model

    def opt_wrapper(self, wtype, value):
        '''Necessary since optimizer requires two components: type and learning rate'''

        if wtype == 'sgd':
            return keras.optimizers.SGD(learning_rate=value)
        if wtype == 'rmsprop':
            return keras.optimizers.RMSprop(learning_rate=value)
        if wtype == 'adagrad':
            return keras.optimizers.Adagrad(learning_rate=value)
        if wtype == 'adadelta':
            return keras.optimizers.Adadelta(learning_rate=value)
        if wtype == 'adam':
            return keras.optimizers.Adam(learning_rate=value)
        if wtype == 'adamax':
            return keras.optimizers.Adamax(learning_rate=value)
        if wtype == 'nadam':
            return keras.optimizers.Nadam(learning_rate=value)

    def reg_wrapper(self, wtype, value):
        '''Necessary since regularizer requires two components: type and learning rate'''

        if wtype == 'l2':
            return l2(value)
        if wtype == 'l1':
            return l1(value)

    def fit(self, hp, model, x, y, **kwargs):
        '''Model fitting'''
        return model.fit(x, y, **kwargs)
