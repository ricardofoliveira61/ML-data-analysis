from Deep_learning.submodels import dense_submodel, gat_submodel
import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Dense
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from inspect import getmembers


class DenseModel:
    """
    Multi-input model for drug response prediction.
    Uses a dense subnetwork for both cell line and drug features.

    Parameters
    ----------
    expr_dim : int
        Dimension of the cell line feature space.
    drug_dim : int
        Dimension of the drug feature space.
    expr_hlayers_sizes : str
        String representation of a list of integers. Each integer represents the size of a hidden layer in the cell line subnetwork.
    drug_hlayers_sizes : str
        String representation of a list of integers. Each integer represents the size of a hidden layer in the drug subnetwork.
    predictor_hlayers_sizes : str
        String representation of a list of integers. Each integer represents the size of a hidden layer in the final network.
    initializer : str
        Initializer for the weights of the dense layers.
    hidden_activation : str
        Activation function for the hidden layers.
    l1 : float
        L1 regularization factor for the final network.
    l2 : float
        L2 regularization factor for the final network.
    input_dropout : float
        Dropout rate for the input layer.
    hidden_dropout : float
        Dropout rate for the hidden layers.
    optimizer : str
        Optimizer for the model.
    learn_rate : float
        Learning rate for the optimizer.

    Returns
    -------
    keras.Model
        Multi-input model for drug response prediction.  
    """
    def __init__(self, expr_dim=None, drug_dim=None, expr_hlayers_sizes='[10]', drug_hlayers_sizes='[10]',
                          predictor_hlayers_sizes='[10]', initializer='he_normal', hidden_activation='relu', l1=0,
                          l2=0, input_dropout=0, hidden_dropout=0, optimizer='Adam', learning_rate=0.001):
        self.cell_line_feature_size = expr_dim
        self.drug_feature_size = drug_dim
        self.expr_hlayers_sizes = expr_hlayers_sizes
        self.drug_hlayers_sizes = drug_hlayers_sizes
        self.predictor_hlayers_sizes = predictor_hlayers_sizes
        self.initializer = initializer
        self.hidden_activation = hidden_activation
        self.l1 = l1
        self.l2 = l2
        self.input_dropout = input_dropout
        self.hidden_dropout = hidden_dropout
        self.optimizer = optimizer
        self.learn_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        # Define input layers
        cell_line_input = Input(shape=(self.cell_line_feature_size,), name='cell_line_input')
        drug_input = Input(shape=(self.drug_feature_size,), name='drug_input')

        # Define subnetworks for cell line and drug
        cell_line_subnet = dense_submodel(cell_line_input)
        drug_subnet = dense_submodel(drug_input)

        # Concatenate the outputs of the attended subnetworks
        concat = Concatenate(name = 'input_layer_concat')([cell_line_subnet, drug_subnet])

        # Final network for IC50 prediction
        final_network = dense_submodel(concat, hlayers_sizes=self.predictor_hlayers_sizes,
	                             l1_regularization=self.l1, l2_regularization=self.l2,
	                             hidden_activation=self.hidden_activation, input_dropout=0,
	                             hidden_dropout=self.hidden_dropout)
        
        final_network = Dense(1, activation='linear', name='ic50_prediction_dense_output')(final_network)


        # Create the model
        model = tf.keras.Model(inputs=[cell_line_input, drug_input], outputs=final_network)
        
        
# Define optimizer
        opt_class = dict(getmembers(optimizers))[self.optimizer]
        if self.optimizer == 'SGD':
            opt = opt_class(learning_rate=self.learn_rate, decay=1e-6, momentum=0.9, nesterov=True)
        else:
            opt = opt_class(learning_rate=self.learn_rate)

        # Compile the model
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae'])

        return model

    def summary(self):
        self.model.summary()

    def train(self, cell_line_train_data, drug_train_data, train_labels, epochs, batch_size):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint('../trained_models/dense_model.keras', monitor='val_loss', save_best_only=True)

        self.history = self.model.fit(
            {'cell_line_input': cell_line_train_data, 'drug_input': drug_train_data},
            {'ic50_prediction_dense_output': train_labels},
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping, model_checkpoint]
        )

    def evaluate(self, cell_line_test_data, drug_test_data, test_labels):
        return self.model.evaluate(
            {'cell_line_input': cell_line_test_data, 'drug_input': drug_test_data},
            {'ic50_prediction_dense_output': test_labels}
        )

    def predict(self, new_cell_line_data, new_drug_data):
        return self.model.predict({'cell_line_input': new_cell_line_data, 'drug_input': new_drug_data})
    
    

def expr_drug_gat_model(expr_dim=None, drug_dim=None, expr_hlayers_sizes='[10]', drug_n_atom_features=30,
                        drug_gat_layers='[64, 64]', drug_num_attention_heads=8, drug_concat_heads=True,
                        drug_residual_connection=True, drug_dropout=0.5, predictor_hlayers_sizes='[10]',
                        initializer='he_normal', hidden_activation='relu', l1=0, l2=0, input_dropout=0,
                        hidden_dropout=0, optimizer='Adam', learn_rate=0.001):
	"""Build a multi-input deep learning model with separate feature-encoding subnetworks for expression and drug
	inputs. The expression subnetwork uses fully-connected layers and the drug subnetwork is a GAT."""
	expr_input = Input(shape=expr_dim, name='expr')
	drugA_nodes_input = Input(shape=(None, drug_n_atom_features), name='drugA_atom_feat')
	drugA_adjacency_input = Input(shape=(None, None), name='drugA_adj')

	expr = dense_submodel(expr_input, hlayers_sizes=expr_hlayers_sizes, l1_regularization=l1, l2_regularization=l2,
	                      hidden_activation=hidden_activation, input_dropout=input_dropout,
	                      hidden_dropout=hidden_dropout)
	drug_submodel = gat_submodel(n_atom_features=drug_n_atom_features, gat_layers=drug_gat_layers,
	                             n_attention_heads=drug_num_attention_heads, concat_heads=drug_concat_heads,
	                             residual=drug_residual_connection, dropout_rate=drug_dropout)
	drugA = drug_submodel([drugA_nodes_input, drugA_adjacency_input])

	concat = concatenate([expr, drugA])

	main_branch = dense_submodel(concat, hlayers_sizes=predictor_hlayers_sizes,
	                             l1_regularization=l1, l2_regularization=l2,
	                             hidden_activation=hidden_activation, input_dropout=0,
	                             hidden_dropout=hidden_dropout)
	# Add output layer
	output = Dense(1, activation='linear', kernel_initializer=initializer, name='output')(main_branch)

	# create Model object
	model = tf.keras.Model(
		inputs=[expr_input, drugA_nodes_input, drugA_adjacency_input],
		outputs=[output])

	# Define optimizer
	opt_class = dict(getmembers(optimizers))[optimizer]
	if optimizer == 'SGD':
		opt = opt_class(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)
	else:
		opt = opt_class(lr=learn_rate)

	# Compile model
	model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae'])

	return model