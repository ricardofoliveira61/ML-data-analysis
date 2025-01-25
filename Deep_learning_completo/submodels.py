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

def gat_submodel(n_atom_features, gat_layers='[64, 64]', n_attention_heads=8, concat_heads=True, residual=False,
                 dropout_rate=0.5, l2=0):
	"""Build a Graph Attention Network (GAT) (Velickovic et al, 2018) submodel."""
	gat_layers = literal_eval(gat_layers)
	regularizer = l1_l2(l1=0, l2=l2)
	nodes_input = Input(shape=(None, n_atom_features))
	adjacency_input = Input(shape=(None, None))
	node_feats = nodes_input
	for n_channels in gat_layers:
		x = GATConv(n_channels, activation='elu', attn_heads=n_attention_heads, concat_heads=concat_heads,
		            dropout_rate=dropout_rate, kernel_initializer='he_normal',
		            kernel_regularizer=regularizer, attn_kernel_regularizer=regularizer,
		            bias_regularizer=regularizer)([node_feats, adjacency_input])
		if residual:  # add a drug_residual_connection connection (as in the implementation of GATPredictor in DGL LifeSci)
			if concat_heads:
				res_feats = Dense(n_channels * n_attention_heads)(
					node_feats)  # according to the implementation of residual connections for GATConv in DGL
			else:
				res_feats = Dense(n_channels)(node_feats)
			x = add([x, res_feats])  # drug_residual_connection connection
		x = Dropout(dropout_rate)(x)
		node_feats = x
	x = GlobalSumPool()(x)

	submodel = tf.keras.Model(inputs=[nodes_input, adjacency_input], outputs=[x], name='drug_gat_submodel')

	return submodel