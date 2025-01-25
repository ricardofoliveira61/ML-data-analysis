import tensorflow as tf
from tensorflow import math
from deepchem.models.layers import Highway, DTNNEmbedding
from tensorflow.keras.layers import AlphaDropout, Activation, Dense, Dropout, BatchNormalization, Input, Conv1D, Lambda, Concatenate, add, LSTM, Bidirectional
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU, PReLU
from ast import literal_eval
from tensorflow.keras.regularizers import l1_l2
from inspect import getmembers
from tensorflow.keras.metrics import MeanSquaredError, RootMeanSquaredError
from spektral.layers import GCNConv, GlobalSumPool, GATConv


def dense_submodel(input_layer, hlayers_sizes='[10]', l1_regularization=0, l2_regularization=0,
                   hidden_activation='relu', input_dropout=0, hidden_dropout=0):
	"""
	Build a dense (fully-connected) subnetwork.
	
	Parameters
	----------
	input_layer : keras.layers.Layer
        Input layer of the subnetwork.
    hlayers_sizes : str
        String representation of a list of integers. Each integer represents the number of neurons in a hidden layer.
    l1_regularization : float
        L1 regularization factor.
    l2_regularization : float
        L2 regularization factor.
    hidden_activation : str
        Activation function to use in the hidden layers.
    input_dropout : float
        Dropout rate for the input layer.
    hidden_dropout : float
        Dropout rate for the hidden layers.
		
    Returns
    -------
    keras.layers.Layer
        Output layer of the subnetwork.
		"""
	hlayers_sizes = literal_eval(hlayers_sizes)  # because hlayers_sizes was passed as a string

	if hidden_activation == 'selu':
		# selu must be used with the LecunNormal initializer and AlphaDropout instead of normal Dropout
		initializer = 'lecun_normal'
		dropout = AlphaDropout
		batchnorm = False
	else:
		initializer = 'he_normal'
		dropout = Dropout
		batchnorm = True

	if input_dropout > 0:
		x = dropout(rate=input_dropout)(input_layer)
	else:
		x = input_layer

	for i in range(len(hlayers_sizes)):
		x = Dense(units=hlayers_sizes[i], kernel_initializer=initializer,
		          kernel_regularizer=l1_l2(l1=l1_regularization, l2=l2_regularization))(x)
		if hidden_activation.lower() == 'leakyrelu':
			x = LeakyReLU()(x)
		elif hidden_activation.lower() == 'prelu':
			x = PReLU()(x)
		else:
			x = Activation(hidden_activation)(x)

		if batchnorm:
			x = BatchNormalization()(x)

		if hidden_dropout > 0:
			x = dropout(rate=hidden_dropout)(x)

	return x
