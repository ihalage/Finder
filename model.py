import numpy as np
import warnings
warnings.filterwarnings("ignore")
from spektral.layers.ops import scatter_mean, scatter_max, scatter_sum, scatter_min#, unsorted_segment_softmax
from spektral.layers import GlobalAvgPool, GlobalMaxPool, CrystalConv, MessagePassing
from global_attn_pool import GlobalAttnAvgPool
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.constraints import max_norm, min_max_norm

from io import StringIO
import sys


# import random as python_random
# np.random.seed(123)
# python_random.seed(123)
# tf.random.set_seed(1234)

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
	- `activation`: activation function;
	- `use_bias`: bool, add a bias vector to the output;
	- `kernel_initializer`: initializer for the weights;
	- `bias_initializer`: initializer for the bias vector;
	- `kernel_regularizer`: regularization applied to the weights;
	- `bias_regularizer`: regularization applied to the bias vector;
	- `activity_regularizer`: regularization applied to the output;
	- `kernel_constraint`: constraint applied to the weights;
	- `bias_constraint`: constraint applied to the bias vector.
	"""

	def __init__(
		self,
		channels,
		aggregate_type="mean",
		use_edge_predictor=False,
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
			use_edge_predictor=False,
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
		self.use_edge_predictor = use_edge_predictor
		self.use_crystal_structure = use_crystal_structure
		self.attn_heads = attn_heads
		# self.enet = self.ENet()
		self.snet = self.SNet()
		if not self.use_crystal_structure:
			self.edgenet = self.EdgeNet()
		# else:
		# 	self.enet = self.ENet()
		self.dq = self.dk = self.dv = int(self.channels/self.attn_heads)
		self.qnet = self.QNet(self.dq)
		self.knet = self.KNet(self.dk)
		self.vnet = self.VNet(self.dv)
		# self.fnet = self.FNet()

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
		self.w_internal = layers.Dense(self.channels, activation='relu') # ///activation="relu", was here with good results
		self.w_o = layers.Dense(self.channels, **layer_kwargs)#//activation="relu", , kernel_constraint=min_max_norm(min_value=1e-40, max_value=1.0)
		self.built = True


	def ENet(self):
		model = tf.keras.Sequential([
		layers.Dense(128, activation="relu", name="layer5", kernel_regularizer=regularizers.l2(1e-6)),#/rev
		layers.Dense(64, activation="relu", name="layer6", kernel_regularizer=regularizers.l2(1e-6)),#//revert
		layers.Dense(1, name="layer7"),
		# layers.BatchNormalization()
		# layers.Dense(2, activation='softmax', name="layer5")
		])
		return model

	def QNet(self, dq):
		model = tf.keras.Sequential([
		layers.Dense(dq, name="layer1", kernel_regularizer=regularizers.l2(1e-8), kernel_constraint=min_max_norm(min_value=1e-30, max_value=1.0)),
		# layers.Dense(64, activation="relu", name="layer2", kernel_regularizer=regularizers.l2(1e-6)),
		# layers.Dense(1, name="layer5")
		])
		return model
	def KNet(self, dk):
		model = tf.keras.Sequential([
		layers.Dense(dk, name="layer1", kernel_regularizer=regularizers.l2(1e-8), kernel_constraint=min_max_norm(min_value=1e-30, max_value=1.0)),
		# layers.Dense(64, activation="relu", name="layer2", kernel_regularizer=regularizers.l2(1e-6)),
		# layers.Dense(1, name="layer5")
		])
		return model
	def VNet(self, dv):
		model = tf.keras.Sequential([
		layers.Dense(dv, name="layer1", kernel_regularizer=regularizers.l2(1e-8), kernel_constraint=min_max_norm(min_value=1e-30, max_value=2.0)),
		# layers.Dense(64, activation="relu", name="layer2", kernel_regularizer=regularizers.l2(1e-6)),
		# layers.Dense(1, name="layer5")
		])
		return model

	def EdgeNet(self):
		model = tf.keras.Sequential([
		# layers.Dense(256, activation="relu", name="layer1", kernel_regularizer=regularizers.l2(1e-6)),#
		# layers.Dense(1024, activation="relu", name="layer2", kernel_regularizer=regularizers.l2(1e-6)),#
		# layers.Dense(1024, activation="relu", name="layer3", kernel_regularizer=regularizers.l2(1e-6)),#
		# layers.Dense(512, activation="relu", name="layer4", kernel_regularizer=regularizers.l2(1e-6)),#
		# layers.Conv1D(1, 3, activation='relu', padding='same'),
		# layers.Flatten(),
		layers.Dense(128, activation="relu", name="layer5", kernel_regularizer=regularizers.l2(1e-6)),#/rev
		layers.Dense(64, activation="relu", name="layer6", kernel_regularizer=regularizers.l2(1e-6)),#//revert
		layers.Dense(1, name="layer7"),
		# layers.BatchNormalization()
		])
		return model

	def SNet(self):
		model = tf.keras.Sequential([
			# layers.BatchNormalization(),
		layers.Dense(128, activation="relu", name="layer4", kernel_regularizer=regularizers.l2(1e-6)),#/rev
		layers.Dense(64, activation="relu", name="layer2", kernel_regularizer=regularizers.l2(1e-6)),
		layers.Dense(self.channels, name="layer5")])
		return model
	def FNet(self):
		model = tf.keras.Sequential([
			# layers.BatchNormalization(),
		# layers.Dense(128, activation="relu", name="layer4", kernel_regularizer=regularizers.l2(1e-6)),
		# layers.Dense(64, activation="relu", name="layer2", kernel_regularizer=regularizers.l2(1e-6)),
		layers.Dense(self.channels, activation="sigmoid", name="layer5")])
		return model

	def unsorted_segment_softmax(self, x, indices, n_nodes=None):
	    n_nodes = tf.reduce_max(indices) + 1 if n_nodes is None else n_nodes
	    e_x = tf.exp(
	        x - tf.gather(tf.math.unsorted_segment_max(x, indices, n_nodes), indices)
	    )
	    # e_x = tf.exp(x)
	    e_x /= tf.gather(
	        tf.math.unsorted_segment_sum(e_x, indices, n_nodes) + 1e-9, indices
	    )
	    return e_x

	def attention(self, x_i, x_j, indices, n_nodes=None, attn_heads=1):
		n_nodes = tf.reduce_max(indices) + 1 if n_nodes is None else n_nodes

		# head_list = []
		 
		# for head in range(attn_heads):
			
		Q = self.qnet(x_i)
		Key = self.knet(x_j)
		V = self.vnet(x_j)

		# QK = tf.exp(Q*Key/np.sqrt(self.dk))
		x = Q*Key/np.sqrt(self.dk)
		QK = tf.exp(
            x - tf.gather(tf.math.unsorted_segment_max(x, indices, n_nodes), indices)
        )
		# tf.debugging.check_numerics(QK, "QK have nan!")
		# ssum = tf.math.unsorted_segment_sum(QK, indices, n_nodes)
		# tf.debugging.check_numerics(ssum, "QK_ssum have nan!")
		QK_soft = QK/ tf.gather(
	        tf.math.unsorted_segment_sum(QK, indices, n_nodes) + 1e-9, indices
	    	)
		# tf.debugging.check_numerics(QK_soft, "QK_soft have nan!")
		C = QK_soft*V
		# head_list.append(C)
		# tf.debugging.check_numerics(C, "C have nan!")
		return C
		# return self.w_o(K.concatenate(head_list, axis=-1))

		# y, idx, splits = tf.unique_with_counts(indices)
		# tf.gather(tf.split(K, splits, 0), indices)

	def sigmoid(self, x, a=0.5):
		return 1/(1+tf.math.exp(-x*a+1))
	def tanh(self, x, a=0.5):
		return (tf.math.exp(a*x)-1)/(tf.math.exp(a*x)+1)
	def standardise(self, x):
		return (x-0.001)/1000.0

	# def clipped_relu(self, x, a=20):
	# 	if x<0:
	# 		return 0
	# 	elif x<a:
	# 		return x
	# 	else:
	# 		return 0.99*x




	def message(self, x, e=None):
		x_i = self.get_i(x)
		x_j = self.get_j(x)

		to_concat = [x_i, x_j]
		# neighbors = K.stack(to_concat, axis=-1)
		# neighbors_concat = K.concatenate(to_concat, axis=-1)
		neighbors_mean = (x_i+x_j)/2.0
		if not self.use_crystal_structure:
			global_attr = scatter_mean(neighbors_mean, self.index_i, self.n_nodes)
			_,_,nodes_list = tf.unique_with_counts(self.index_i)
			# gat = tf.repeat(global_attr, nodes_list, axis=-2)
			gat = tf.gather(global_attr, self.index_i)
			neighbors = K.concatenate([neighbors_mean, gat], axis=-1)
			# neighbors = K.concatenate([x_i, x_j, gat], axis=-1)

			edge = self.edgenet(neighbors)
			# l = K.print_tensor(edge, message='edge attribute = ')

			# c=tf.print(edge, summarize=-1)

			# one_string = tf.strings.format("{}\n", edge, summarize=-1)
			# tf.io.write_file('edge_attribute', one_string, name=None)


		if self.use_edge_predictor:
			eloss  = K.mean(K.abs(edge - e))
			# eloss  = K.mean(K.abs(tf.gather(edge, tf.random.shuffle(tf.range(tf.shape(edge)[0]), seed=10)) - tf.gather(e, tf.random.shuffle(tf.range(tf.shape(e)[0]), seed=10))))
			self.add_loss(eloss)
			# l = K.print_tensor(edge, message='\n\nedge = ')
			# l = K.print_tensor(e, message='e = ')
			# print([edge.eval(), e.eval()])
		# l = K.print_tensor(edge[:10], message='\n\nedge = ')

		# eij = self.enet(neighbors_mean)	## uncomment
		# aij = self.unsorted_segment_softmax(eij, self.index_i, self.n_nodes)	# uncomment
		# edge_aij = self.unsorted_segment_softmax(edge, self.index_i, self.n_nodes)	# uncomment
		aij = self.attention(x_i, x_j, self.index_i, self.n_nodes, attn_heads=1)
		# tf.debugging.check_numerics(aij, "aij weights have nan!")

		# to_concat_new = [x_i, x_j, edge]	# uncomment for actual model
		# to_concat_new = [neighbors_mean, edge]	# uncomment
		if self.use_crystal_structure:
			'''
			new additions
			'''
			# global_attr = scatter_mean(neighbors_mean, self.index_i, self.n_nodes)
			# gat = tf.gather(global_attr, self.index_i)
			# neighbors = K.concatenate([neighbors_mean, gat, e], axis=-1)
			# edge = self.edgenet(neighbors)
			# edge = self.enet(neighbors)
			# to_concat_new = [neighbors_mean, edge]
			# l = K.print_tensor(e, message='e = ')
			# print('\n\nneighbours mean shape:', neighbors_mean.shape)
			# print('E shape:', e.shape)
			to_concat_new = [neighbors_mean, e]#tf.keras.activations.relu(e, max_value=4)
			# l = K.print_tensor(to_concat_new, message='to_concat_new = ')
		else:
			to_concat_new = [neighbors_mean, edge]


		z = K.concatenate(to_concat_new, axis=-1)
		# z = K.batch_normalization(z, mean=0, var=0.5, gamma=1, beta=0)
		# l = K.print_tensor(z, message='Z = ')
		# tf.debugging.check_numerics(z, "Z  have nan!")
		output = (aij*self.snet(z))# * (edge_aij* self.fnet(z))
		# snet_w = self.snet(z)
		# tf.debugging.check_numerics(snet_w, "SNet weights have nan!")
		# l = K.print_tensor(snet_w, message='SNet weights = ')
		output = K.batch_normalization(output, mean=0, var=0.5, gamma=1, beta=0)
		# output = K.normalize_batch_in_training(output, gamma=1, beta=0, reduction_axes=[-1])

		return output#, softmaxed
	def aggregate(self, messages):
		# m, s = messages
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
		# l = K.print_tensor(self.w_internal.weights, message='w_internal = ')
		# updated_x = self.w_internal(x) + embeddings
		# X = tf.strings.format("{}\n", updated_x, summarize=-1)
		# tf.io.write_file('global_attribute', X, name=None)
		
		return self.w_internal(x) + embeddings

	@property
	def config(self):
		return {"channels": self.channels}




class Net(Model):
	def __init__(self,
				channels=200,
				n_out=1,
				robust=True,
				aggregate_type="mean",
				use_edge_predictor=False,
				use_crystal_structure=False):
		super().__init__()
		self.conv1 = GraphLayer(channels, aggregate_type=aggregate_type, 
								use_edge_predictor=use_edge_predictor,
								use_crystal_structure=use_crystal_structure, activation='relu')
		self.conv2 = GraphLayer(channels, aggregate_type=aggregate_type,
								use_edge_predictor=use_edge_predictor,
								use_crystal_structure=use_crystal_structure, activation='relu')
		# self.conv3 = GraphLayer(channels, aggregate_type=aggregate_type,
		# 						use_edge_predictor=use_edge_predictor,
		# 						use_crystal_structure=use_crystal_structure, activation='relu')
		# self.global_pool1 = GlobalAvgPool()
		# self.global_pool2 = GlobalAvgPool()
		self.global_pool1 = GlobalAttnAvgPool(channels=channels)
		self.global_pool2 = GlobalAttnAvgPool(channels=channels)
		self.bn1 = layers.BatchNormalization()
		self.bn2 = layers.BatchNormalization()
		self.conv1d1 = layers.Conv1D(64, 3, activation='relu', padding='same')
		self.dense1 = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-6))
		self.drop1 = layers.Dropout(rate=0.2)
		self.dense2 = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(1e-6))
		self.drop2 = layers.Dropout(rate=0.2)
		self.dense3 = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(1e-6))
		self.dense4 = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-6))
		n_out = 2*n_out if robust else n_out
		self.dense5 = layers.Dense(int(n_out), activation='linear')

	def call(self, inputs):
		x, a, e, i = inputs
		x1 = self.conv1([x, a, e])
		# x1 = self.bn1(x1)
		x2 = self.conv2([x1, a, e])
		# x2 = self.conv3([x2, a, e])
		# x2 = self.bn2(x2)
		x1_gp = self.global_pool1([x1, i])
		x2_gp = self.global_pool2([x2, i])
		x = tf.expand_dims(x2_gp, axis=-1)
		x = self.conv1d1(x)
		x3 = layers.Flatten()(x)
		x = self.dense1(x3)
		# x = self.drop1(x)
		x = self.dense2(layers.concatenate([x, x1_gp]))
		# x = self.drop2(x)
		x = self.dense3(layers.concatenate([x, x2_gp]))
		x = self.dense4(layers.concatenate([x, x3]))
		output = self.dense5(x)
		return output