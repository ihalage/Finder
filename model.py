import numpy as np
import warnings
warnings.filterwarnings("ignore")
from spektral.layers.ops import scatter_mean, scatter_max, scatter_sum, scatter_min
from spektral.layers import MessagePassing
from global_attn_pool import GlobalAttnAvgPool
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.constraints import min_max_norm

class GraphLayer(MessagePassing):
	r"""
	**Input**
	- Node features of shape `(n_nodes, n_node_features)`;
	- Binary adjacency matrix of shape `(n_nodes, n_nodes)`.
	- Edge features of shape `(num_edges, n_edge_features)`.
	**Output**
	- Node features with the same shape of the input, but the last dimension
	changed to `channels`.
	**Arguments**
	- `channels`: integer, number of output channels;
	- `aggregate_type`: permutation invariant function to aggregate messages
	- `use_crystal_structure`: use crystal structure details (crystal graph)
	- `activation`: activation function;
	- `use_bias`: bool, add a bias vector to the output;
	- `kernel_initializer`: initializer for the weights;
	- `bias_initializer`: initializer for the bias vector;
	- `kernel_regularizer`: regularization applied to the weights;
	- `bias_regularizer`: regularization applied to the bias vector;
	- `activity_regularizer`: regularization applied to the output;
	- `kernel_constraint`: constraint applied to the weights;
	- `bias_constraint`: constraint applied to the bias vector.
	- `attn_heads`: number of attention heads
	"""

	def __init__(
		self,
		channels,
		aggregate_type="mean",
		use_crystal_structure=False,
		activation=None,
		use_bias=True,
		kernel_initializer="glorot_uniform",
		bias_initializer="zeros",
		kernel_regularizer=None,
		bias_regularizer=None,
		activity_regularizer=None,
		kernel_constraint=None,
		bias_constraint=None,
		attn_heads=1,
		**kwargs
	):
		super().__init__(
			aggregate_type=aggregate_type,
			use_crystal_structure=False,
			activation=activation,
			use_bias=use_bias,
			kernel_initializer=kernel_initializer,
			bias_initializer=bias_initializer,
			kernel_regularizer=kernel_regularizer,
			bias_regularizer=bias_regularizer,
			activity_regularizer=activity_regularizer,
			kernel_constraint=kernel_constraint,
			bias_constraint=bias_constraint,
			attn_heads=attn_heads,
			**kwargs
		)
		self.channels = channels
		self.aggregate_type = aggregate_type
		self.use_crystal_structure = use_crystal_structure
		self.attn_heads = attn_heads
		self.snet = self.SNet()
		if not self.use_crystal_structure:	# we need the edge predicting network if crystal structure is not used
			self.edgenet = self.EdgeNet()

		self.dq = self.dk = self.dv = int(self.channels/self.attn_heads)
		self.qnet = self.QNet(self.dq)
		self.knet = self.KNet(self.dk)
		self.vnet = self.VNet(self.dv)

	def build(self, input_shape):
		assert len(input_shape) >= 2
		layer_kwargs = dict(
			kernel_initializer=self.kernel_initializer,
			bias_initializer=self.bias_initializer,
			kernel_regularizer=self.kernel_regularizer,
			bias_regularizer=self.bias_regularizer,
			kernel_constraint=self.kernel_constraint,
			bias_constraint=self.bias_constraint,
			dtype=self.dtype,
		)
		self.w_internal = layers.Dense(self.channels, activation='relu')
		self.w_o = layers.Dense(self.channels, **layer_kwargs)
		self.built = True

	def QNet(self, dq):
		model = tf.keras.Sequential([
		layers.Dense(dq, name="layer1", kernel_regularizer=regularizers.l2(1e-8), kernel_constraint=min_max_norm(min_value=1e-30, max_value=1.0)), # weight clipping to avoid exploding gradients
		])
		return model
	def KNet(self, dk):
		model = tf.keras.Sequential([
		layers.Dense(dk, name="layer1", kernel_regularizer=regularizers.l2(1e-8), kernel_constraint=min_max_norm(min_value=1e-30, max_value=1.0)),
		])
		return model
	def VNet(self, dv):
		model = tf.keras.Sequential([
		layers.Dense(dv, name="layer1", kernel_regularizer=regularizers.l2(1e-8), kernel_constraint=min_max_norm(min_value=1e-30, max_value=2.0)),
		])
		return model

	def EdgeNet(self):
		model = tf.keras.Sequential([
		layers.Dense(128, activation="relu", name="layer5", kernel_regularizer=regularizers.l2(1e-6)),
		layers.Dense(64, activation="relu", name="layer6", kernel_regularizer=regularizers.l2(1e-6)),
		layers.Dense(1, name="layer7"),
		])
		return model

	def SNet(self):
		model = tf.keras.Sequential([
		layers.Dense(128, activation="relu", name="layer4", kernel_regularizer=regularizers.l2(1e-6)),
		layers.Dense(64, activation="relu", name="layer2", kernel_regularizer=regularizers.l2(1e-6)),
		layers.Dense(self.channels, name="layer5")])
		return model

	def unsorted_segment_softmax(self, x, indices, n_nodes=None):
	    n_nodes = tf.reduce_max(indices) + 1 if n_nodes is None else n_nodes
	    e_x = tf.exp(
	        x - tf.gather(tf.math.unsorted_segment_max(x, indices, n_nodes), indices)
	    )
	    e_x /= tf.gather(
	        tf.math.unsorted_segment_sum(e_x, indices, n_nodes) + 1e-9, indices
	    )
	    return e_x

	def attention(self, x_i, x_j, indices, n_nodes=None, attn_heads=1):
		n_nodes = tf.reduce_max(indices) + 1 if n_nodes is None else n_nodes
	
		Q = self.qnet(x_i)
		Key = self.knet(x_j)
		V = self.vnet(x_j)

		x = Q*Key/np.sqrt(self.dk)
		QK = tf.exp(
            x - tf.gather(tf.math.unsorted_segment_max(x, indices, n_nodes), indices)
        )

		QK_soft = QK/ tf.gather(
	        tf.math.unsorted_segment_sum(QK, indices, n_nodes) + 1e-9, indices
	    	)
		C = QK_soft*V

		return C

	def message(self, x, e=None):
		x_i = self.get_i(x)
		x_j = self.get_j(x)

		neighbors_mean = (x_i+x_j)/2.0
		
		if self.use_crystal_structure: # crystal graph. Edge attribute is available as distance
			to_concat = [neighbors_mean, e]
		else: # integer formula graph. Edge attribute is predicted during training
			global_attr = scatter_mean(neighbors_mean, self.index_i, self.n_nodes)
			_,_,nodes_list = tf.unique_with_counts(self.index_i)
			global_at = tf.gather(global_attr, self.index_i)
			neighbors = K.concatenate([neighbors_mean, global_at], axis=-1)
			
			edge = self.edgenet(neighbors)
			to_concat = [neighbors_mean, edge]

		z = K.concatenate(to_concat, axis=-1)
		aij = self.attention(x_i, x_j, self.index_i, self.n_nodes, attn_heads=1) # self-attention

		output = (aij*self.snet(z))
		output = K.batch_normalization(output, mean=0, var=0.5, gamma=1, beta=0)

		return output

	def aggregate(self, messages):
		if self.aggregate_type=='mean':
			return scatter_mean(messages, self.index_i, self.n_nodes)
		elif self.aggregate_type=='sum':
			return scatter_sum(messages, self.index_i, self.n_nodes)
		elif self.aggregate_type=='max':
			return scatter_max(messages, self.index_i, self.n_nodes)
		elif self.aggregate_type=='min':
			return scatter_min(messages, self.index_i, self.n_nodes)
		else:
			warnings.warn("Specified aggregate function not available. Using mean as the aggregate function ...")
			return scatter_mean(messages, self.index_i, self.n_nodes)

	def update(self, embeddings, x=None):
		return self.w_internal(x) + embeddings

	@property
	def config(self):
		return {"channels": self.channels}




class Finder(Model):
	def __init__(self,
				channels=200,
				n_out=1,
				robust=True,
				aggregate_type="mean",
				use_crystal_structure=False):
		super().__init__()
		self.conv1 = GraphLayer(channels, aggregate_type=aggregate_type, 
								use_crystal_structure=use_crystal_structure, activation='relu')
		self.conv2 = GraphLayer(channels, aggregate_type=aggregate_type,
								use_crystal_structure=use_crystal_structure, activation='relu')

		self.global_pool1 = GlobalAttnAvgPool(channels=channels)
		self.global_pool2 = GlobalAttnAvgPool(channels=channels)

		self.conv1d1 = layers.Conv1D(64, 3, activation='relu', padding='same')
		self.dense1 = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-6))
		# self.drop1 = layers.Dropout(rate=0.2)
		self.dense2 = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(1e-6))
		# self.drop2 = layers.Dropout(rate=0.2)
		self.dense3 = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(1e-6))
		self.dense4 = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-6))
		n_out = 2*n_out if robust else n_out
		self.dense5 = layers.Dense(int(n_out), activation='linear')

	def call(self, inputs):
		x, a, e, i = inputs
		x1 = self.conv1([x, a, e])
		x2 = self.conv2([x1, a, e])

		x1_gp = self.global_pool1([x1, i])
		x2_gp = self.global_pool2([x2, i])

		x = tf.expand_dims(x2_gp, axis=-1)
		x = self.conv1d1(x)

		x3 = layers.Flatten()(x)
		x = self.dense1(x3)
		x = self.dense2(layers.concatenate([x, x1_gp]))
		x = self.dense3(layers.concatenate([x, x2_gp]))
		x = self.dense4(layers.concatenate([x, x3]))
		output = self.dense5(x)
		return output