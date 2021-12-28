import numpy as np
import pandas as pd
import scipy.sparse as sp
import json
import warnings

import cv2
from pymatgen import Composition, Element, Structure
from pymatgen.analysis.structure_analyzer import  VoronoiConnectivity

from spektral.data import Dataset, DisjointLoader, Graph, BatchLoader
from spektral.transforms.normalize_adj import NormalizeAdj
from spektral.layers.ops import scatter_mean, scatter_max, scatter_sum
from matminer.featurizers.composition import ElementFraction


import itertools
from utils import ElemProp

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from spektral.layers import GCSConv, GlobalAvgPool, GlobalMaxPool, GlobalAttentionPool, GlobalAttnSumPool, SAGPool, CrystalConv
from spektral.layers.pooling import TopKPool
from spektral.layers import GCNConv, MessagePassing, GATConv, GraphSageConv
from spektral.models.gcn import GCN
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError
from spektral.layers.convolutional import gcn_conv
from spektral.layers import ECCConv, GlobalSumPool
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense

from spektral.layers.convolutional.message_passing import MessagePassing


# tf.config.run_functions_eagerly(True)
# physical_devices = tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.39
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


"""
A sub-class to loead custom dataset
"""

class DataLoader(Dataset):
	def __init__(self, 
				data_path,
				num_targets=1,
				is_train=True,
				embedding_path='data/embeddings/',
				embedding_type='mat2vec',
				max_coord_no=40,
				n_classes=32,
				**kwargs):

		self.data_path = data_path
		self.num_targets = num_targets
		self.is_train = is_train
		self.embedding_path = embedding_path
		self.embedding_type = embedding_type
		self.max_coord_no = max_coord_no
		self.ElemProp = ElemProp()
		self.efCls = ElementFraction()
		self.n_classes = n_classes
		self.znet = tf.keras.models.load_model('saved_models/best_model_cnn_Z.h5')
		self.dismatnet =  tf.keras.models.load_model('saved_models/best_model_cnn_DisMat.h5')
		super().__init__(**kwargs)

	def getEmbeddings(self):
		with open(self.embedding_path+self.embedding_type+'-embedding.json') as file:
			data = json.load(file)
			return dict(data)

	def parse_formulae(self, data_path=None, data=None):
		if data_path is not None:
			df = pd.read_csv(data_path, keep_default_na = False)
		if data is not None:
			df = data
		# df = df[df.Z <= self.n_classes]
		df_a = df.apply(lambda x: self.efCls.featurize(Composition(x.formula)), axis=1, result_type='expand')
		df_featurized = pd.concat([df, df_a], axis='columns')
		X = df_featurized.drop(['ID', 'formula', 'integer_formula', 'nsites', 'Z', 'target', 'cif', 'nelements', 'is_inert_gas', 'num_atoms'], axis=1, errors='ignore')

		return np.expand_dims(X, axis=-1)
	
	def read(self):	# n_nodes = n_element type
		embeddings = self.getEmbeddings()
		df = pd.read_pickle(self.data_path)#.iloc[:1000]
		
		# num_sites = 
		df['nelements'] = df['formula'].apply(lambda x: len(Composition(x).elements))
		df['is_inert_gas'] = df['formula'].apply(lambda x: any(i in Composition(x).elements for i in [Element('He'), Element('Ne'), Element('Ar'), Element('Kr'), Element('Xe')]))
		df['num_atoms'] = df['formula'].apply(lambda x: Composition(Composition(x).get_integer_formula_and_factor()[0]).num_atoms)
		df = df[(df.nelements>1) & (df.is_inert_gas==False) & (df.num_atoms < 500)]
		print(df.info())
		if 'formula' in df.iloc[0].tolist():
			df = df.iloc[1:]
		graph_list = []	# final list of graphs to be returned
		# df.reset_index(drop=True, inplace=True)
		# X = self.parse_formulae(data=df)
		# print(X.shape)
		# print(self.znet.summary())
		# ## get NN predictions ##
		# Z = np.argmax(self.znet.predict([X]), axis=1)+1
		# dismat_list = np.squeeze(self.dismatnet.predict([X]))*10.0
		# # print(Z)
		# # print(dismat_list.shape)
		# # print(Z.shape, dismat_list.shape)

		def make_graph(row):
			idx = row.name
			# print('this is index: ', idx)
			# formula = row['integer_formula']
			formula = row['formula']
			# print(formula)
			# label = row[1:3001]
			label = row[2:3002]
			# cif_str = row['cif']
			# s = Structure.from_str(cif_str, fmt='cif')
			# comp = Composition(formula)
			# all_sites = [site.specie for site in s.sites]
			# if self.is_train:
			# 	total_atoms = Composition(s.formula).num_atoms
			# 	int_form_atoms = Composition(formula).num_atoms
			# 	z = int(total_atoms/int_form_atoms)
			# else:
			# z = Z[idx]
			comp = Composition(Composition(formula).get_integer_formula_and_factor()[0])
			elem_dict = comp.get_el_amt_dict()
			all_sites = []
			for e, n in elem_dict.items():
				el = Element(e)
				all_sites.extend([el]*int(n))

			N = len(all_sites)

			# comp = Composition(Composition(formula).get_integer_formula_and_factor()[0])
			# elem_dict = comp.get_el_amt_dict()
			# metals = []
			# # metalloids = []
			# non_metals = []
			# for e,n in elem_dict.items():
			# 	el = Element(e)
			# 	if el.is_metal:
			# 		metals.extend([el]*int(n))
			# 	# elif el.is_metalloid:
			# 	# 	metalloids.extend([el]*int(n))
			# 	else:
			# 		non_metals.extend([el]*int(n))
			
			#### Node feature vector (X) ####
			X = []
			# all_elems = metals + metalloids + non_metals
			# all_elems = metals + non_metals
			# id_pad = 0
			# global_f = self.efCls.featurize(Composition(formula))
			# print(global_f)
			for e in all_sites:
				X.append(embeddings[e.name])
				# print(X[0])
				# id_pad = (id_pad+1)/1000.0
			# X = np.array(X)

			# #### Adjacency matrix (A) and Edge attributes (E) ####
			# distance_matrix = s.distance_matrix
			# if self.is_train:
			# 	distance_matrix = s.distance_matrix
			# else:
			# 	distance_matrix = dismat_list[idx]
			# 	distance_matrix = cv2.resize(distance_matrix, dsize=(N, N))

			# A = np.where(distance_matrix<5, 1, 0)
			# np.fill_diagonal(A, 0)
			A = 1-np.eye((len(all_sites)))
			# ids = np.where(A==1)
			# E = np.expand_dims(np.ones(len(ids[0])), axis=-1)
			# E = []
			x, y = np.where(A==1)
			# E = np.expand_dims(distance_matrix[x, y], axis=-1)
			E = np.expand_dims(A[x, y], axis=-1)
			# E = global_f
			# for i, j in zip(x, y):
			# 	if (all_sites[i].is_metal and not(all_sites[j].is_metal)):	# ionic bond. A metal and a non_metal is bonded
			# 		r_m = list(self.ElemProp.get_ionic_radius(all_sites[i].name).values())[-1]
			# 		r_nm = list(self.ElemProp.get_ionic_radius(all_sites[j].name).values())[0]
			# 		bond_length = r_m + r_nm
			# 		E.append([bond_length])
			# 		# e_neg_diff = abs(all_sites[i].X - all_sites[j].X)
			# 		# E.append([bond_length, e_neg_diff, 1, 0, 0])
			# 	elif (all_sites[j].is_metal and not(all_sites[i].is_metal)): # ionic bond
			# 		r_m = list(self.ElemProp.get_ionic_radius(all_sites[j].name).values())[-1]
			# 		r_nm = list(self.ElemProp.get_ionic_radius(all_sites[i].name).values())[0]
			# 		bond_length = r_m + r_nm
			# 		E.append([bond_length])
			# 		# e_neg_diff = abs(all_sites[j].X - all_sites[i].X)
			# 		# E.append([bond_length, e_neg_diff, 1, 0, 0])
			# 	elif (not(all_sites[i].is_metal) and not(all_sites[j].is_metal)): # covalent bond
			# 		r_nm1 = self.ElemProp.get_covalent_radius(all_sites[i].name)
			# 		r_nm2 = self.ElemProp.get_covalent_radius(all_sites[j].name)
			# 		bond_length = r_nm1 + r_nm2
			# 		E.append([bond_length])
			# 		# e_neg_diff = abs(all_sites[i].X - all_sites[j].X)
			# 		# E.append([bond_length, e_neg_diff, 0, 1, 0])
			# 	else: # metallic bond
			# 		r_m1 = self.ElemProp.get_metallic_radius(all_sites[i].name)
			# 		r_m2 = self.ElemProp.get_metallic_radius(all_sites[j].name)
			# 		bond_length = r_m1 + r_m2
			# 		E.append([bond_length])
			# 		# e_neg_diff = abs(all_sites[i].X - all_sites[j].X)
			# 		# E.append([bond_length, e_neg_diff, 0, 0, 1])

			# A = np.zeros((len(all_sites), len(all_sites)))
			# v = VoronoiConnectivity(s, cutoff=6)
			# neighbors = np.array(v.get_connections())
			# A[neighbors[:,0].astype(int), neighbors[:,1].astype(int)] = 1
			# E = np.expand_dims(neighbors[:,2], axis=-1)



			# n_bonds = [0] * len(all_elems)	# track the number of bonds for all elements in 'all_elems'. Max No. bonds per element is 12

			# A = np.zeros((len(all_elems), len(all_elems)))	# Adjacency matrix
			# E = []	# Edge features

			# m = len(metals)
			# n = len(non_metals)
			# m_i = list(range(len(metals)))
			# n_i = list(range(len(metals), len(all_elems)))
			# if metals and non_metals:
			# 	# print("\n\nMetal - Non_metal: ", formula)
			# 	ids = np.array(list(itertools.product(m_i, n_i)))
			# 	x = ids[:,0]
			# 	y = ids[:,1]

			# 	# First add all bonds (metals and non_metals are densely connected)
			# 	A[x,y] = 1
			# 	A[y,x] = 1	# due to symmetry

			# 	A[m_i, 0:-min(n, self.max_coord_no)] = 0
			# 	A[0:-min(n, self.max_coord_no), m_i] = 0	# due to symmetry

			# 	A[n_i, min(self.max_coord_no,m):] = 0
			# 	A[min(self.max_coord_no,m):, n_i] = 0	# due to symmetry

			# 	# check if there are any unbonded elements
			# 	unbonded_ids = np.where(A.sum(axis=1)==0)[0]
			# 	if len(unbonded_ids)>0:
			# 		if m>self.max_coord_no and n>self.max_coord_no:
			# 			A[range(self.max_coord_no, min(m, 2*self.max_coord_no)), -min(n, 2*self.max_coord_no):-self.max_coord_no] = 1
			# 			A[-min(n, 2*self.max_coord_no):-self.max_coord_no, range(self.max_coord_no, min(m, 2*self.max_coord_no))] = 1
			# 		elif n>self.max_coord_no:
			# 			A[unbonded_ids[:min(n-self.max_coord_no, self.max_coord_no-m)], -min(self.max_coord_no, n):] = 1	# ///// check this line what if m~self.max_coord_no
			# 			A[-min(self.max_coord_no, n):, unbonded_ids[:min(n-self.max_coord_no, self.max_coord_no-m)]] = 1
			# 			if (n-self.max_coord_no) > (self.max_coord_no-m):
			# 				A[unbonded_ids[min(n-self.max_coord_no, self.max_coord_no-m):], -n:-min(self.max_coord_no, n):] = 1	# this might violate 12 max_coord rule for large formula
			# 				A[-n:-min(self.max_coord_no, n):, unbonded_ids[min(n-self.max_coord_no, self.max_coord_no-m):]] = 1
			# 			# A[-min(self.max_coord_no, n):, unbonded_ids[:min(n-self.max_coord_no, self.max_coord_no-m)]] = 1
			# 			A[unbonded_ids, unbonded_ids] = 0
			# 		else:
			# 			A[unbonded_ids, 0:min(m, self.max_coord_no)] = 1	# metals don't usually exceed 2*max_coord
			# 			A[0:min(m, self.max_coord_no), unbonded_ids] = 1

			# 	# ## densly connected graph ##
			# 	A = 1-np.eye(len(all_elems))
			# 	####
			# 	# Create edge attributes
			# 	x, y = np.where(A==1)
			# 	for i, j in zip(x, y):
			# 		if (i in m_i and j in n_i):	# ionic bond. A metal and a non_metal is bonded
			# 			r_m = list(self.ElemProp.get_ionic_radius(all_elems[i].name).values())[-1]
			# 			r_nm = list(self.ElemProp.get_ionic_radius(all_elems[j].name).values())[0]
			# 			bond_length = r_m + r_nm
			# 			e_neg_diff = abs(all_elems[i].X - all_elems[j].X)
			# 			E.append([bond_length, e_neg_diff, 1, 0, 0])
			# 		if (j in m_i and i in n_i): # ionic bond
			# 			r_m = list(self.ElemProp.get_ionic_radius(all_elems[j].name).values())[-1]
			# 			r_nm = list(self.ElemProp.get_ionic_radius(all_elems[i].name).values())[0]
			# 			bond_length = r_m + r_nm
			# 			e_neg_diff = abs(all_elems[j].X - all_elems[i].X)
			# 			E.append([bond_length, e_neg_diff, 1, 0, 0])
			# 		if (i in n_i and j in n_i): # covalent bond
			# 			r_nm1 = self.ElemProp.get_covalent_radius(all_elems[i].name)
			# 			r_nm2 = self.ElemProp.get_covalent_radius(all_elems[j].name)
			# 			bond_length = r_nm1 + r_nm2
			# 			e_neg_diff = abs(all_elems[i].X - all_elems[j].X)
			# 			E.append([bond_length, e_neg_diff, 0, 1, 0])
			# 		if (i in m_i and j in m_i): # metallic bond
			# 			r_m1 = self.ElemProp.get_metallic_radius(all_elems[i].name)
			# 			r_m2 = self.ElemProp.get_metallic_radius(all_elems[j].name)
			# 			bond_length = r_m1 + r_m2
			# 			e_neg_diff = abs(all_elems[i].X - all_elems[j].X)
			# 			E.append([bond_length, e_neg_diff, 0, 0, 1])

			# elif non_metals:
			# 	# print("\n\nNon_metal only: ", formula)
			# 	# A = 1-np.eye(n)
			# 	if n<=self.max_coord_no:
			# 		A = 1-np.eye(n)	# all elements are densely connected
			# 	elif self.max_coord_no<= (n/2.0):
			# 		# fill = np.ones((self.max_coord_no, self.max_coord_no))
			# 		i=0
			# 		while((i+1)*self.max_coord_no<=(n/2.0)):
			# 			if i==0:
			# 				A[i*self.max_coord_no:(i+1)*self.max_coord_no, -(i+1)*self.max_coord_no:]=1
			# 				A[-(i+1)*self.max_coord_no:, i*self.max_coord_no:(i+1)*self.max_coord_no]=1
			# 			else:
			# 				A[i*self.max_coord_no:(i+1)*self.max_coord_no, -(i+1)*self.max_coord_no:-i*self.max_coord_no]=1
			# 				A[-(i+1)*self.max_coord_no:-i*self.max_coord_no, i*self.max_coord_no:(i+1)*self.max_coord_no]=1
			# 			i+=1
			# 		unbonded_ids = np.where(A.sum(axis=1)==0)[0]
			# 		if len(unbonded_ids)>1:	# else we have one unbonded element
			# 			A[unbonded_ids, unbonded_ids[::-1]]=1
			# 	else:
			# 		A[0:int(n/2), -int(n/2):]=1
			# 		A[-int(n/2):, 0:int(n/2)]=1

			# 	# ## densly connected graph ##
			# 	A = 1-np.eye(len(all_elems))
			# 	####
			# 	# create edge attributes
			# 	x, y = np.where(A==1)
			# 	for i, j in zip(x, y):	# all covalent bonds
			# 		r_nm1 = self.ElemProp.get_covalent_radius(all_elems[i].name)
			# 		r_nm2 = self.ElemProp.get_covalent_radius(all_elems[j].name)
			# 		bond_length = r_nm1 + r_nm2
			# 		e_neg_diff = abs(all_elems[i].X - all_elems[j].X)
			# 		E.append([bond_length, e_neg_diff, 0, 1, 0])

			# # metals
			# else:
			# 	# print("\n\nMetal only: ", formula)
			# 	if m<=self.max_coord_no:
			# 		A = 1-np.eye(m)	# densly connected
			# 	elif self.max_coord_no<= (m/2.0):
			# 		# fill = np.ones((self.max_coord_no, self.max_coord_no))
			# 		i=0
			# 		while((i+1)*self.max_coord_no<=(m/2.0)):
			# 			if i==0:
			# 				A[i*self.max_coord_no:(i+1)*self.max_coord_no, -(i+1)*self.max_coord_no:]=1
			# 				A[-(i+1)*self.max_coord_no:, i*self.max_coord_no:(i+1)*self.max_coord_no]=1
			# 			else:
			# 				A[i*self.max_coord_no:(i+1)*self.max_coord_no, -(i+1)*self.max_coord_no:-i*self.max_coord_no]=1
			# 				A[-(i+1)*self.max_coord_no:-i*self.max_coord_no, i*self.max_coord_no:(i+1)*self.max_coord_no]=1
			# 			i+=1
			# 		unbonded_ids = np.where(A.sum(axis=1)==0)[0]
			# 		if len(unbonded_ids)>1:	# else we have one unbonded element
			# 			A[unbonded_ids, unbonded_ids[::-1]]=1
			# 	else:
			# 		A[0:int(m/2), -int(m/2):]=1
			# 		A[-int(m/2):, 0:int(m/2)]=1

			# 	# ## densly connected graph ##
			# 	A = 1-np.eye(len(all_elems))
			# 	####
			# 	# create edge attributes
			# 	x, y = np.where(A==1)
			# 	for i, j in zip(x, y):	# all metallic bonds
			# 		r_m1 = self.ElemProp.get_metallic_radius(all_elems[i].name)
			# 		r_m2 = self.ElemProp.get_metallic_radius(all_elems[j].name)
			# 		bond_length = r_m1 + r_m2
			# 		e_neg_diff = abs(all_elems[i].X - all_elems[j].X)
			# 		E.append([bond_length, e_neg_diff, 0, 0, 1])
			# is_symmetric = (A==A.T).all()
			# print(is_symmetric)
			# if not is_symmetric:
			# 	return graph_list
			X = np.array(X)

			# print(distance_matrix)
			# print(formula, z)
			# print(X.shape)
			# print(A.shape)
			# print(distance_matrix.shape, '\n\n')
			

			
			A = sp.csr_matrix(A)
			E = np.array(E)
			Y = np.array(label, dtype=np.float)


			graph = Graph(x=X, a=A, e=E, y=Y)
			graph_list.append(graph)
		df.apply(make_graph, axis=1)
		return graph_list


class CrystalConv_ENet(MessagePassing):
	r"""
	A crystal graph convolutional layer from the paper
	> [Crystal Graph Convolutional Neural Networks for an Accurate and
	Interpretable Prediction of Material Properties](https://arxiv.org/abs/1710.10324)<br>
	> Tian Xie and Jeffrey C. Grossman
	**Mode**: single, disjoint, mixed.
	**This layer expects a sparse adjacency matrix.**
	This layer computes:
	$$
		\x_i' = \x_i + \sum\limits_{j \in \mathcal{N}(i)} \sigma \left( \z_{ij}
		\W^{(f)} + \b^{(f)} \right) \odot \g \left( \z_{ij} \W^{(s)} + \b^{(s)}
		\right)
	$$
	where \(\z_{ij} = \x_i \| \x_j \| \e_{ji} \), \(\sigma\) is a sigmoid
	activation, and \(g\) is the activation function (defined by the `activation`
	argument).
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
		aggregate="mean",
		activation=None,
		use_bias=True,
		kernel_initializer="glorot_uniform",
		bias_initializer="zeros",
		kernel_regularizer=None,
		bias_regularizer=None,
		activity_regularizer=None,
		kernel_constraint=None,
		bias_constraint=None,
		**kwargs
	):
		super().__init__(
			aggregate=aggregate,
			activation=activation,
			use_bias=use_bias,
			kernel_initializer=kernel_initializer,
			bias_initializer=bias_initializer,
			kernel_regularizer=kernel_regularizer,
			bias_regularizer=bias_regularizer,
			activity_regularizer=activity_regularizer,
			kernel_constraint=kernel_constraint,
			bias_constraint=bias_constraint,
			**kwargs
		)
		self.channels = channels
		self.enet = self.ENet()
		self.snet = self.SNet()
		# self.updatenet = self.updateNet()

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
		# self.dense_f = Dense(self.channels, activation="sigmoid", **layer_kwargs)
		# self.dense_s = Dense(self.channels, activation=self.activation, **layer_kwargs)
		# self.dense_f = tf.keras.Sequential([
		# 							layers.Dense(128, activation='relu'),
		# 							# layers.Dense(512, activation='relu'),
		# 							layers.Dense(self.channels, activation="sigmoid", **layer_kwargs)])
		# self.dense_s = tf.keras.Sequential([
		# 							layers.Dense(128, activation='relu'),
		# 							# layers.Dense(512, activation='relu'),
		# 							layers.Dense(self.channels, activation=self.activation, **layer_kwargs)])

		

		self.built = True


	def ENet(self):
		model = tf.keras.Sequential([
		# layers.Conv1D(2, 3, activation="relu", name="layer1"),
		# layers.Conv1D(128, 3, activation="relu", name="layer2"),
		# layers.Conv1D(256, 3, activation="relu", name="layer3"),
		# layers.Flatten(),
		layers.Dense(256, activation="relu", name="layer4", kernel_regularizer=regularizers.l2(1e-6)),
		layers.Dense(128, activation="relu", name="layer2", kernel_regularizer=regularizers.l2(1e-6)),
		# layers.Dense(128, activation="relu", name="layer3"),
		# layers.Dense(256, activation="relu", name="layer4"),
		layers.Dense(1, name="layer5")])
		return model

	def SNet(self):
		model = tf.keras.Sequential([
		# layers.Conv1D(2, 3, activation="relu", name="layer1"),
		# layers.Conv1D(128, 3, activation="relu", name="layer2"),
		# layers.Conv1D(256, 3, activation="relu", name="layer3"),
		# layers.Flatten(),
		layers.Dense(256, activation="relu", name="layer4", kernel_regularizer=regularizers.l2(1e-6)),
		layers.Dense(128, activation="relu", name="layer2", kernel_regularizer=regularizers.l2(1e-6)),
		# layers.Dense(128, activation="relu", name="layer3"),
		# layers.Dense(256, activation="relu", name="layer4"),
		layers.Dense(self.channels, name="layer5")])
		return model

	def updateNet(self):
		model = tf.keras.Sequential([
		# layers.Conv1D(2, 3, activation="relu", name="layer1"),
		# layers.Conv1D(128, 3, activation="relu", name="layer2"),
		# layers.Conv1D(256, 3, activation="relu", name="layer3"),
		# layers.Flatten(),
		# layers.Dense(32, activation="relu", name="layer4"),
		# layers.Dense(64, activation="relu", name="layer2"),
		# layers.Dense(128, activation="relu", name="layer3"),
		# layers.Dense(256, activation="relu", name="layer4"),
		layers.Dense(self.channels, name="layer5")])
		return model


	def message(self, x, e=None):
		x_i = self.get_i(x)
		x_j = self.get_j(x)
		# print(x_i.shape, x_j.shape, e.shape)

		
		# if e is not None:
		# 	to_concat += [e]
		# agg = scatter_mean(x_j, self.index_i, self.n_nodes)
		# if K.mean(x_i, axis=1) >= K.mean(x_j, axis=1):

		to_concat = [x_i, x_j]
		# neighbors = K.stack(to_concat, axis=-1)
		# neighbors = K.concatenate(to_concat, axis=-1)
		# neighbors = x_i+x_j
		neighbors = (x_i+x_j)/2.0
		global_attr = scatter_max(neighbors, self.index_i, self.n_nodes)
		_,_,nodes_list = tf.unique_with_counts(self.index_i)
		gat = tf.repeat(global_attr, nodes_list, axis=-2)
		# print(x.shape, neighbors.shape, gat.shape, global_attr.shape, self.n_nodes)
		neighbors = K.concatenate([neighbors, gat], axis=-1)


		E = self.enet(neighbors)
		# print(x_i.shape, x_j.shape, e.shape, neighbors.shape, E.shape)
		to_concat_new = [x_i, x_j, E]
		z = K.concatenate(to_concat_new, axis=-1)
		# output = self.dense_s(z) * self.dense_f(z)
		output = self.snet(z) #* self.dense_f(z)

		return output
	def aggregate(self, messages):
		# print(self.n_nodes)
		# if self.aggregation_type=="mean":
			# print(scatter_mean(messages, self.index_i, self.n_nodes).shape)
		return scatter_mean(messages, self.index_i, self.n_nodes)

	def update(self, embeddings, x=None):
		return x + embeddings

	@property
	def config(self):
		return {"channels": self.channels}





class Net(Model):
	def __init__(self):
		super().__init__()
		# self.conv1 = ECCConv(32, activation="relu")
		# self.conv2 = ECCConv(32, activation="relu")
		# self.conv1 = MPCNN()
		# self.conv1 = CrystalConv(200, activation='relu')
		self.conv1 = CrystalConv_ENet(200, activation='relu')
		self.conv2 = CrystalConv_ENet(200, activation='relu')
		# self.conv2 = CrystalConv(200, activation='relu')
		# self.topkpool = TopKPool(ratio=0.5)
		# self.sagpool = SAGPool(ratio=0.5)
		# self.conv2 = MPCNN()
		self.global_pool1 = GlobalAvgPool()
		self.global_pool2 = GlobalAvgPool()
		self.conv1d1 = layers.Conv1D(64, 3, activation='relu', padding='same')
		self.conv1d2 = layers.Conv1D(128, 3, activation='relu', padding='same')
		self.conv1d3 = layers.Conv1D(256, 3, activation='relu', padding='same')
		self.conv1d4 = layers.Conv1D(256, 3, activation='relu', padding='same')
		self.dense1 = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-6))
		self.dense2 = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(1e-6))
		self.dense3 = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(1e-6))
		self.dense4 = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-6))
		self.dense5 = Dense(n_out, activation='linear')

	def call(self, inputs):
		x, a, e, i = inputs
		x1 = self.conv1([x, a, e])
		x2 = self.conv2([x1, a, e])
		# x, a, i = self.topkpool([x, a, i])
		# x, a, i = self.sagpool([x, a, i])
		# x = self.conv2([x, a, e])
		x1_gp = self.global_pool2([x1, i])
		x2_gp = self.global_pool1([x2, i])
		x = tf.expand_dims(x2_gp, axis=-1)
		x = self.conv1d1(x)
		# output = self.conv1d2(output)
		# output = self.conv1d3(output)
		# output = self.conv1d4(output)
		x3 = layers.Flatten()(x)
		x = self.dense1(x3)
		x = self.dense2(layers.concatenate([x, x1_gp]))
		x = self.dense3(layers.concatenate([x, x2_gp]))
		x = self.dense4(layers.concatenate([x, x3]))
		output = self.dense5(x)
		# x = self.global_pool1([x2, i])
		# x = tf.expand_dims(x, axis=-1)
		# x = self.conv1d1(x)

		# x3 = layers.Flatten()(x)
		# x = self.dense1(x3)
		# x = self.dense2(x)
		# x = self.dense3(x)
		# x = self.dense4(layers.concatenate([x, x3]))
		# output = self.dense5(x)




		return output




if __name__=='__main__':
	# dataset_tr = DataLoader('data/databases/MP_formation_energy/train_with_Z.csv', is_train=True)
	# dataset_val = DataLoader('data/databases/MP_formation_energy/test_with_Z.csv', is_train=False)
	dataset_tr = DataLoader('data/databases/difunc_data/RealX_train.pkl')
	dataset_val = DataLoader('data/databases/difunc_data/RealX_val.pkl')
	dataset_te = DataLoader('data/databases/difunc_data/RealX_test.pkl')
	# dataset_te = DataLoader('data/databases/MP_formation_energy/test.csv')
	# loader.read()
	# print(dataset[0])
	learning_rate = 3e-4  # Learning rate
	epochs = 250  # Number of training epochs
	batch_size = 128  # Batch size

		
	# Parameters
	F = dataset_tr.n_node_features  # Dimension of node features
	N = dataset_tr.n_nodes
	S = dataset_tr.n_edge_features  # Dimension of edge features
	n_out = dataset_tr.n_labels  # Dimension of the target
	print("NUMBER ", F, N, S, n_out)

	# # Train/test split
	# idxs = np.random.permutation(len(dataset))
	# split = int(0.8 * len(dataset))
	# idx_tr, idx_te = np.split(idxs, [split])
	# dataset_tr, dataset_te = dataset[idx_tr], dataset[idx_te]

	loader_tr = DisjointLoader(dataset_tr, batch_size=batch_size, epochs=epochs)
	loader_val = DisjointLoader(dataset_val, batch_size=batch_size, epochs=1)
	loader_te = DisjointLoader(dataset_te, batch_size=batch_size, epochs=1, shuffle=False)
	
	# print(len(val_in))

	# loader_tr = DisjointLoader(dataset_tr, batch_size=batch_size, epochs=epochs)
	# loader_val = DisjointLoader(dataset_val, batch_size=batch_size, epochs=epochs)
	# loader_te = DisjointLoader(dataset_te, batch_size=batch_size, epochs=1)
	# print(loader_tr.load())

	X_in = Input(shape=(F,), name="X_in")
	A_in = Input(shape=(N,), sparse=True)
	E_in = Input(shape=(), name='Edges')
	I_in = Input(shape=(), name="segment_ids_in", dtype=tf.int32)

	# dropout = 0.5
	# channels = 8
	# l2_reg = 2.5e-4
	# n_attn_heads = 8
	# do_1 = tf.keras.layers.Dropout(dropout)(X_in)
	# gc_1 = GATConv(
	#     channels,
	#     attn_heads=n_attn_heads,
	#     concat_heads=True,
	#     dropout_rate=dropout,
	#     activation="elu",
	#     kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
	#     attn_kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
	#     bias_regularizer=tf.keras.regularizers.l2(l2_reg),
	# )([do_1, A_in])
	# do_2 = tf.keras.layers.Dropout(dropout)(gc_1)
	# gc_2 = GATConv(
	#     n_out,
	#     attn_heads=1,
	#     concat_heads=False,
	#     dropout_rate=dropout,
	#     activation="relu",
	#     kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
	#     attn_kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
	#     bias_regularizer=tf.keras.regularizers.l2(l2_reg),
	# )([do_2, A_in])

	# X = MessagePassing( aggregate="max")([X_in, A_in])
	# # X, A, I_1 = TopKPool(ratio=0.5)([X, A_in, I_in])
	# # X = MessagePassing()([X, A])
	# X = GlobalAvgPool()(X)
	# output = Dense(n_out)(X)



	# X_1 = GCSConv(32, activation="relu")([X_in, A_in])
	# # X_1 = MessagePassing()([X_in, A_in])
	# X_1, A_1, I_1 = TopKPool(ratio=0.5)([X_1, A_in, I_in])
	# X_2 = GCSConv(32, activation="relu")([X_1, A_1])
	# X_2, A_2, I_2 = TopKPool(ratio=0.5)([X_2, A_1, I_1])
	# X_3 = GCSConv(32, activation="relu")([X_2, A_2])
	# X_3, A_3, I_3 = TopKPool(ratio=0.5)([X_3, A_2, I_2])
	# X_4 = GCSConv(32, activation="relu")([X_3, A_3])
	# X_4 = GlobalAvgPool()([X_4, I_3])
	# output = Dense(n_out, activation="linear")(X_4)


	# X_1 = MPCNN()([X_in, A_in])
	# # X_1, A_1, I_1 = TopKPool(ratio=0.5)([X_1, A_in, I_in])
	# X_2 = GlobalAvgPool()([X_1])
	# output = Dense(n_out, activation="linear")(X_2)

	# X_1 = MPCNN()([X_in, A_in])
	# # X_1 = MessagePassing()([X_in, A_in])
	# X_1, A_1, I_1 = TopKPool(ratio=0.5)([X_1, A_in, I_in])
	# X_2 = MPCNN()([X_1, A_1])
	# X_2, A_2, I_2 = TopKPool(ratio=0.5)([X_2, A_1, I_1])
	# X_3 = MPCNN()([X_2, A_2])
	# X_3, A_3, I_3 = TopKPool(ratio=0.5)([X_3, A_2, I_2])
	# X_4 = MPCNN()([X_3, A_3])
	# X_4 = GlobalAvgPool()([X_4, I_3])
	# output = Dense(n_out, activation="linear")(X_4)


	# ################################################################################
	# # BUILD MODEL
	# ################################################################################
	# X_in = Input(shape=(F,), name="X_in")
	# A_in = Input(shape=(None,), sparse=True, name="A_in")
	# E_in = Input(shape=(S,), name="E_in")
	# I_in = Input(shape=(), name="segment_ids_in", dtype=tf.int32)

	# X_1 = ECCConv(32, activation="relu")([X_in, A_in, E_in])
	# X_2 = ECCConv(32, activation="relu")([X_1, A_in, E_in])
	# X_3 = GlobalSumPool()([X_2, I_in])
	# output = Dense(n_out)(X_3)

	# # Build model
	# model = Model(inputs=[X_in, A_in, I_in], outputs=output)
	# opt = Adam(lr=learning_rate)
	# loss_fn = MeanAbsoluteError()
	# model.compile(loss=loss_fn,
	# 				optimizer=opt, #tfa.optimizers.LAMB(learning_rate=lr),
	# 				metrics='mae')

	model = Net()
	optimizer = Adam(learning_rate)
	loss_fn = MeanAbsoluteError()

	# model.compile(loss=loss_fn,
	# 				optimizer=optimizer, #tfa.optimizers.LAMB(learning_rate=lr),
	# 				metrics='mae')
	# model.fit(loader_tr.load(), steps_per_epoch=loader_tr.steps_per_epoch, validation_data=loader_val.load(), validation_steps=loader_val.steps_per_epoch, epochs=epochs, batch_size=batch_size)
	# print(model.summary())

	################################################################################
	# Fit model
	################################################################################
	@tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
	def train_step(inputs, target):
		with tf.GradientTape() as tape:
			predictions = model(inputs, training=True)
			loss = loss_fn(target, predictions) + sum(model.losses)
		gradients = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(gradients, model.trainable_variables))
		return loss


	step = loss = 0
	validation_data = list(loader_val)
	epoch_no = 1
	best_val_loss = 100
	for batch in loader_tr:
		step += 1
		loss += train_step(*batch)
		if step == loader_tr.steps_per_epoch:
			val_loss = 0
			# loader_val = DisjointLoader(dataset_val, batch_size=batch_size, epochs=1)
			for batch_val in validation_data:
				val_inputs, val_targets = batch_val
				val_predictions = model(val_inputs, training=False)
				val_loss += loss_fn(val_targets, val_predictions)
				# val_loss_total+=val_loss
			step = 0
			print('\nEpoch: ', epoch_no)
			print("Training Loss: {} ....... Validation loss: {}\n".format(loss / loader_tr.steps_per_epoch, val_loss / loader_val.steps_per_epoch))
			epoch_no+=1
			# print("Validation Loss: {}".format(val_loss / loader_val.steps_per_epoch))
			loss = 0

	################################################################################
	# Evaluate model
	################################################################################
	print("Testing model")
	loss = 0
	target_list = []
	predictions_list = []
	for batch in loader_te:
		inputs, target = batch
		predictions = model(inputs, training=False)
		# preds = model.predict(inputs)
		# print(preds.shape)
		# print(predictions.numpy(), target.numpy())
		loss += loss_fn(target, predictions)
	loss /= loader_te.steps_per_epoch
	print("Done. Test loss: {}".format(loss))
	loader_te = DisjointLoader(dataset_te, batch_size=batch_size, epochs=1, shuffle=False)
	preds = model.predict(loader_te.load())
	# print(preds)
	np.save('difunc_test_pred.npy', preds) 
	# df_preds = pd.DataFrame({"predictions": preds.flatten()})
	# df_preds.to_csv("OQMD__Eform_preds_MPCNN.csv", header=None, index=False)

	print(preds.shape)


	
	# model.fit(loader_tr.load(), steps_per_epoch=loader_tr.steps_per_epoch, validation_data=loader_val.load(), validation_steps=loader_val.steps_per_epoch, epochs=epochs, batch_size=batch_size)


	# ################################################################################
	# # FIT MODEL
	# ################################################################################
	# @tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
	# def train_step(inputs, target):
	#     with tf.GradientTape() as tape:
	#         predictions = model(inputs, training=True)
	#         loss = loss_fn(target, predictions)
	#         loss += sum(model.losses)
	#     gradients = tape.gradient(loss, model.trainable_variables)
	#     opt.apply_gradients(zip(gradients, model.trainable_variables))
	#     return loss


	# print("Fitting model")
	# current_batch = 0
	# model_loss = 0
	# for batch in loader_tr:
	#     outs = train_step(*batch)

	#     model_loss += outs
	#     current_batch += 1
	#     if current_batch == loader_tr.steps_per_epoch:
	#         print("Loss: {}".format(model_loss / loader_tr.steps_per_epoch))
	#         model_loss = 0
	#         current_batch = 0

	# ################################################################################
	# # EVALUATE MODEL
	# ################################################################################
	# print("Testing model")
	# model_loss = 0
	# for batch in loader_te:
	#     inputs, target = batch
	#     predictions = model(inputs, training=False)
	#     model_loss += loss_fn(target, predictions)
	# model_loss /= loader_te.steps_per_epoch
	# print("Done. Test loss: {}".format(model_loss))