import numpy as np
import pandas as pd
import scipy.sparse as sp
import json
import warnings

import cv2
from pymatgen import Composition, Element, Structure
from pymatgen.analysis.structure_analyzer import  VoronoiConnectivity
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from spektral.data import Dataset, DisjointLoader, Graph, BatchLoader
from spektral.transforms.normalize_adj import NormalizeAdj
from spektral.layers.ops import scatter_mean, scatter_max, scatter_sum#, unsorted_segment_softmax
from matminer.featurizers.composition import ElementFraction
from sklearn.preprocessing import StandardScaler

import itertools
from utils import ElemProp
from utils import Normalizer, NormalizeTensor

from tensorflow.python.keras.utils import losses_utils
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, regularizers
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from spektral.layers import GCSConv, GlobalAvgPool, GlobalMaxPool, GlobalAttentionPool, GlobalAttnSumPool, SAGPool, CrystalConv
from global_attn_pool import GlobalAttnAvgPool
from spektral.layers.pooling import TopKPool
from spektral.layers import GCNConv, MessagePassing, GATConv, GraphSageConv
from spektral.models.gcn import GCN
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError, RobustLoss, Huber
from spektral.layers.convolutional import gcn_conv
from spektral.layers import ECCConv, GlobalSumPool
# from tensorflow.keras import backend as K
# from tensorflow.keras.layers import Dense

from spektral.layers.convolutional.message_passing import MessagePassing


# tf.config.run_functions_eagerly(True)
# physical_devices = tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.42
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

scaler = Normalizer()



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
		self.soapnet = tf.keras.models.load_model('saved_models/best_model_cnn_SoapNet.h5')
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
		X = df_featurized.drop(['ID', 'formula', 'integer_formula', 'nsites', 'Z', 'target', 'cif', 'nelements', 'is_inert_gas'], axis=1, errors='ignore')

		return np.expand_dims(X, axis=-1)
	
	def read(self):	# n_nodes = n_element type
		embeddings = self.getEmbeddings()
		df = pd.read_csv(self.data_path)#.iloc[:1000]
		
		# num_sites = 
		df['nelements'] = df['formula'].apply(lambda x: len(Composition(x).elements))
		df['is_inert_gas'] = df['formula'].apply(lambda x: any(i in Composition(x).elements for i in [Element('He'), Element('Ne'), Element('Ar'), Element('Kr'), Element('Xe')]))
		df['num_atoms'] = df['formula'].apply(lambda x: Composition(Composition(x).get_integer_formula_and_factor()[0]).num_atoms)
		df = df[(df.nelements>1) & (df.is_inert_gas==False) & (df.num_atoms < 500)]
		print(df.info())
		if 'formula' in df.iloc[0].tolist():
			df = df.iloc[1:]
		graph_list = []	# final list of graphs to be returned
		df.reset_index(drop=True, inplace=True)
		X = self.parse_formulae(data=df)
		if self.is_train:
			scaler.fit(np.array(df['target']))
		df['target'] = scaler.norm(np.array(df['target']))#df['target'].apply(lambda x: len(Composition(x).elements))
		print(X.shape)
		# print(self.znet.summary())
		# ## get NN predictions ##
		# Z = np.argmax(self.znet.predict([X]), axis=1)+1
		dismat_list = np.squeeze(self.dismatnet.predict([X]))#*10.0
		# soaps_list = self.soapnet.predict([X])
		# # print(Z)
		# # print(dismat_list.shape)
		# # print(Z.shape, dismat_list.shape)

		def make_graph(row):
			idx = row.name
			# print('this is index: ', idx)
			# formula = row['integer_formula']
			formula = row['formula']
			# print(formula)
			label = row['target']
			# cs_to_number = {'cubic': 0, 'tetragonal': 1, 'orthorhombic': 2, 'triclinic': 3, 'monoclinic': 4, 'trigonal': 5, 'hexagonal': 6}
			# cs_list = np.array([0, 0, 0, 0, 0, 0, 0])
			cif_str = row['cif']
			s = Structure.from_str(cif_str, fmt='cif')
			# SGA = SpacegroupAnalyzer(s)
			# cs = SGA.get_crystal_system()
			# # s = SGA.get_conventional_standard_structure()
			# cs_list[cs_to_number[cs]]=1
			# label = cs_list
			# label = s.lattice.a


			# comp = Composition(formula)
			# all_sites = [site.specie for site in s.sites]
			# if self.is_train:
			# 	total_atoms = Composition(s.formula).num_atoms
			# 	int_form_atoms = Composition(formula).num_atoms
			# 	z = int(total_atoms/int_form_atoms)
			# else:
			# z = Z[idx]

			# z=row.Z

			full_formula = s.formula
			z = Composition(full_formula).get_integer_formula_and_factor()[1]
			comp_dict = Composition(s.formula).get_el_amt_dict()
			# full_formula = s.formula
			# ax[0].imshow(s.distance_matrix)
			# dm = s.distance_matrix
			sites = [s[l] for l in range(s.num_sites)]
			norms=np.linalg.norm(s.fractional_distance_matrix, ord=2, axis=1)
			j = 0
			sorted_ids = []
			for e,n in comp_dict.items():
				# print(e, n)
				sorted_ids.extend(list((np.argsort(norms[j:j+int(n)])+j)[:int(n/z)]))
				j=j+int(n)
			for k in range(int(s.num_sites/z)):
				s[k] = sites[sorted_ids[k]]
			s.remove_sites(range(int(s.num_sites/z), s.num_sites))
			distance_matrix = s.fractional_distance_matrix

			# all_sites = [s[i] for i in range(s.num_sites)]

			# uncomment for composition based prediction
			comp = Composition(Composition(formula).get_integer_formula_and_factor()[0])
			elem_dict = comp.get_el_amt_dict()
			all_sites = []
			for e, n in elem_dict.items():
				el = Element(e)
				# all_sites.extend([el]*int(n))
				all_sites.extend([[el, n]]*int(n))

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
			# global_f = np.array(global_f).reshape(1,-1)
			# soap_pred = self.soapnet()
			# print(global_f)
			i=0
			for e in all_sites:
				# print(e)
				# X.append(embeddings[e.name])
				
				X.append(embeddings[e[0].name]+[i/100.0, e[1]/100.0])
				i+=1
				if i==e[1]:
					i=0
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

			
			# distance_matrix = dismat_list[idx]
			# distance_matrix = cv2.resize(distance_matrix, dsize=(N, N))
			

			# soaps = soaps_list[idx]
			# A = np.where(distance_matrix<5, 1, 0)
			# np.fill_diagonal(A, 0)
			A = 1-np.eye((len(all_sites)))
			# ids = np.where(A==1)
			# E = np.expand_dims(np.ones(len(ids[0])), axis=-1)
			# E = []
			x, y = np.where(A==1)
			E = np.expand_dims(distance_matrix[x, y], axis=-1)

			# E = np.expand_dims(A[x, y], axis=-1)
			# E = np.vstack([soaps]*int(A.sum()))
			# E = global_f
			# for i, j in zip(x, y):
			# 	if (all_sites[i].is_metal and not(all_sites[j].is_metal)):	# ionic bond. A metal and a non_metal is bonded
			# 		r_m = list(self.ElemProp.get_ionic_radius(all_sites[i].name).values())[-1]
			# 		r_nm = list(self.ElemProp.get_ionic_radius(all_sites[j].name).values())[0]
			# 		bond_length = r_m + r_nm
			# 		# E.append([bond_length])
			# 		e_neg_diff = abs(all_sites[i].X - all_sites[j].X)
			# 		E.append([bond_length, e_neg_diff, 1, 0, 0])
			# 	elif (all_sites[j].is_metal and not(all_sites[i].is_metal)): # ionic bond
			# 		r_m = list(self.ElemProp.get_ionic_radius(all_sites[j].name).values())[-1]
			# 		r_nm = list(self.ElemProp.get_ionic_radius(all_sites[i].name).values())[0]
			# 		bond_length = r_m + r_nm
			# 		# E.append([bond_length])
			# 		e_neg_diff = abs(all_sites[j].X - all_sites[i].X)
			# 		E.append([bond_length, e_neg_diff, 1, 0, 0])
			# 	elif (not(all_sites[i].is_metal) and not(all_sites[j].is_metal)): # covalent bond
			# 		r_nm1 = self.ElemProp.get_covalent_radius(all_sites[i].name)
			# 		r_nm2 = self.ElemProp.get_covalent_radius(all_sites[j].name)
			# 		bond_length = r_nm1 + r_nm2
			# 		# E.append([bond_length])
			# 		e_neg_diff = abs(all_sites[i].X - all_sites[j].X)
			# 		E.append([bond_length, e_neg_diff, 0, 1, 0])
			# 	else: # metallic bond
			# 		r_m1 = self.ElemProp.get_metallic_radius(all_sites[i].name)
			# 		r_m2 = self.ElemProp.get_metallic_radius(all_sites[j].name)
			# 		bond_length = r_m1 + r_m2
			# 		# E.append([bond_length])
			# 		e_neg_diff = abs(all_sites[i].X - all_sites[j].X)
			# 		E.append([bond_length, e_neg_diff, 0, 0, 1])

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
			Y = label


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
		self.edgenet = self.EdgeNet()
		# self.updatenet = self.updateNet()
		# self.qnet = self.QNet()
		# self.knet = self.KNet()
		# self.vnet = self.VNet()

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
		# input_shape = input_shape[0].value
		# shape = (3, input_shape[0][-1]*4, 32)
		# shape = (input_shape[0][-1]*4, 200)
		# self.fu_kernel = self.add_weight(name='kernel', shape=shape,
  #                                     initializer='glorot_uniform')

		

		self.built = True


	def ENet(self):
		model = tf.keras.Sequential([
			# layers.BatchNormalization(),
		# layers.Conv1D(2, 3, activation="relu", name="layer1"),
		# layers.Conv1D(128, 3, activation="relu", name="layer2"),
		# layers.Conv1D(256, 3, activation="relu", name="layer3"),
		# layers.Flatten(),
		layers.Dense(128, activation="relu", name="layer4", kernel_regularizer=regularizers.l2(1e-6)),
		layers.Dense(64, activation="relu", name="layer2", kernel_regularizer=regularizers.l2(1e-6)),
		# layers.Dense(128, activation="relu", name="layer3"),
		# layers.Dense(256, activation="relu", name="layer4"),
		layers.Dense(1, name="layer5")])
		return model

	def EdgeNet(self):
		model = tf.keras.Sequential([
			# layers.BatchNormalization(),
		# layers.Conv1D(2, 3, activation="relu", name="layer1"),
		# layers.Conv1D(128, 3, activation="relu", name="layer2"),
		# layers.Conv1D(256, 3, activation="relu", name="layer3"),
		# layers.Flatten(),
		layers.Dense(128, activation="relu", name="layer4", kernel_regularizer=regularizers.l2(1e-6)),
		layers.Dense(64, activation="relu", name="layer2", kernel_regularizer=regularizers.l2(1e-6)),
		# layers.Dense(128, activation="relu", name="layer3"),
		# layers.Dense(256, activation="relu", name="layer4"),
		layers.Dense(1, name="layer5")])
		return model

	def SNet(self):
		model = tf.keras.Sequential([
			# layers.BatchNormalization(),
		# layers.Conv1D(2, 3, activation="relu", name="layer1"),
		# layers.Conv1D(128, 3, activation="relu", name="layer2"),
		# layers.Conv1D(256, 3, activation="relu", name="layer3"),
		# layers.Flatten(),
		layers.Dense(128, activation="relu", name="layer4", kernel_regularizer=regularizers.l2(1e-6)),
		layers.Dense(64, activation="relu", name="layer2", kernel_regularizer=regularizers.l2(1e-6)),
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

	def QNet(self):
		model = tf.keras.Sequential([layers.Dense(8)])
		return model
	def KNet(self):
		model = tf.keras.Sequential([layers.Dense(200)])
		return model
	def VNet(self):
		model = tf.keras.Sequential([layers.Dense(8)])
		return model

	def unsorted_segment_softmax(self, x, indices, n_nodes=None):
	    """
	    Applies softmax along the segments of a Tensor. This operator is similar
	    to the tf.math.segment_* operators, which apply a certain reduction to the
	    segments. In this case, the output tensor is not reduced and maintains the
	    same shape as the input.
	    :param x: a Tensor. The softmax is applied along the first dimension.
	    :param indices: a Tensor, indices to the segments.
	    :param n_nodes: the number of unique segments in the indices. If `None`,
	    n_nodes is calculated as the maximum entry in the indices plus 1.
	    :return: a Tensor with the same shape as the input.
	    """
	    n_nodes = tf.reduce_max(indices) + 1 if n_nodes is None else n_nodes
	    # e_x = tf.exp(
	    #     x - tf.gather(tf.math.unsorted_segment_max(x, indices, n_nodes), indices)
	    # )
	    e_x = tf.exp(x)
	    # global_attr = tf.math.unsorted_segment_sum(e_x, indices, n_nodes)
	    # _,_,nodes_list = tf.unique_with_counts(self.index_i)
	    # gat = tf.repeat(global_attr, nodes_list, axis=-2)
	    e_x /= tf.gather(
	        tf.math.unsorted_segment_sum(e_x, indices, n_nodes) + 1e-9, indices
	    )
	    return e_x

	def FU(self, x):
		# # x: input features with shape [N,C,H,W]# y_r / y_i is the real / imaginary part of the results of FFT, respectively
		# y_r, y_i = FFT(x) # y_r/y_i: [N,C,H,bW2c+1]
		# y = Concatenate([y_r, y_i], dim=1) # [N,C∗2,H,bW2c+1]
		# y = ReLU(BN(Conv(y))) # [N,C∗2,H,bW2c+1]
		# y_r, y_i = Split(y, dim=1) # y_r/y_i: [N,C,H,bW2c+1]
		# z = iFFT(y_r, y_i) # [N,C,H,W]
		# return z
		x = tf.cast(x, tf.complex64)
		fft = tf.signal.fft(x)
		y_r, y_i = tf.math.real(fft), tf.math.imag(fft)
		y = K.concatenate([y_r, y_i], axis=-1)
		# y = layers.Conv1D(tf.expand_dims(y, axis=-1), 32, strides=1, padding='same')(y)
		# y = K.conv1d(tf.expand_dims(y, axis=-1), self.fu_kernel, strides=1, padding='same')
		y = K.dot(y, self.fu_kernel)
		# y = tf.squeeze(y)
		# y_r, y_i = tf.split(y, 2, axis=1)
		# complex_y = tf.dtypes.complex(y_r, y_i)
		# y = tf.signal.ifft(complex_y)
		# y = tf.cast(y, tf.float32)

		return y



	def message(self, x, e=None):
		x_i = self.get_i(x)
		x_j = self.get_j(x)
		# print(x.shape, x_i.shape, x_j.shape, e.shape)

		
		# if e is not None:
		# 	to_concat += [e]
		# agg = scatter_mean(x_j, self.index_i, self.n_nodes)
		# if K.mean(x_i, axis=1) >= K.mean(x_j, axis=1):

		to_concat = [x_i, x_j]
		# neighbors = K.stack(to_concat, axis=-1)
		# neighbors_concat = K.concatenate(to_concat, axis=-1)
		# neighbors = x_i+x_j
		neighbors_mean = (x_i+x_j)/2.0
		global_attr = scatter_mean(neighbors_mean, self.index_i, self.n_nodes)
		_,_,nodes_list = tf.unique_with_counts(self.index_i)
		# gat = tf.repeat(global_attr, nodes_list, axis=-2)
		gat = tf.gather(global_attr, self.index_i)
		# print(x.shape, neighbors.shape, gat.shape, global_attr.shape, self.n_nodes)
		neighbors = K.concatenate([neighbors_mean, gat], axis=-1)
		# neighbors = tf.signal.rfft(neighbors)
		edge = self.edgenet(neighbors)
		eloss  = K.mean(K.abs(edge - e))
		self.add_loss(eloss)


		eij = self.enet(neighbors_mean)

		aij = self.unsorted_segment_softmax(eij, self.index_i, self.n_nodes)
		# softmaxed = self.unsorted_segment_softmax(x_j, self.index_i, self.n_nodes)

		# Q = self.qnet(x_i)
		# V = self.vnet(x_j)


		# E = self.enet(neighbors)
		
		# E = layers.MultiHeadAttention(num_heads=2, key_dim=2)(query=tf.expand_dims(Q, axis=-1), value=tf.expand_dims(V, axis=-1))
		# E = self.knet(tf.squeeze(E))
		# print(x_i.shape, x_j.shape, e.shape, neighbors.shape, E.shape)
		to_concat_new = [x_i, x_j, edge]
		z = K.concatenate(to_concat_new, axis=-1)
		# z = self.FU(z)
		# output = self.dense_s(z) * self.dense_f(z)
		# print(z.shape)
		output = aij*self.snet(z) #* self.dense_f(z)

		return output#, softmaxed
	def aggregate(self, messages):
		# m, s = messages
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
		self.conv1 = CrystalConv_ENet(202, activation='relu')
		self.conv2 = CrystalConv_ENet(202, activation='relu')
		# self.conv2 = GATConv(200, attn_heads=4, )
		# self.conv3 = CrystalConv(200, activation='relu')
		# self.topkpool = TopKPool(ratio=0.5)
		# self.sagpool = SAGPool(ratio=0.5)
		# self.conv2 = MPCNN()
		# self.global_pool1 = GlobalAvgPool()
		# self.global_pool2 = GlobalAvgPool()
		self.global_pool1 = GlobalAttnAvgPool()
		self.global_pool2 = GlobalAttnAvgPool()
		self.conv1d1 = layers.Conv1D(64, 3, activation='relu', padding='same')
		self.conv1d2 = layers.Conv1D(128, 3, activation='relu', padding='same')
		self.conv1d3 = layers.Conv1D(256, 3, activation='relu', padding='same')
		self.conv1d4 = layers.Conv1D(256, 3, activation='relu', padding='same')
		self.dense1 = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-6))
		self.drop1 = layers.Dropout(rate=0.2)
		self.dense2 = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(1e-6))
		self.drop2 = layers.Dropout(rate=0.2)
		self.dense3 = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(1e-6))
		self.dense4 = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-6))
		self.dense5 = Dense(2*n_out, activation='linear')

	def call(self, inputs):
		x, a, e, i = inputs
		x1 = self.conv1([x, a, e])
		x2 = self.conv2([x1, a, e])
		# x2 = self.conv3([x2, a, e])
		# x, a, i = self.topkpool([x, a, i])
		# x, a, i = self.sagpool([x, a, i])
		# x = self.conv2([x, a, e])
		x1_gp = self.global_pool1([x1, i])
		x2_gp = self.global_pool2([x2, i])
		x = tf.expand_dims(x2_gp, axis=-1)
		x = self.conv1d1(x)
		# output = self.conv1d2(output)
		# output = self.conv1d3(output)
		# output = self.conv1d4(output)
		x3 = layers.Flatten()(x)
		x = self.dense1(x3)
		# x = self.drop1(x)
		x = self.dense2(layers.concatenate([x, x1_gp]))
		# x = self.drop2(x)
		x = self.dense3(layers.concatenate([x, x2_gp]))
		x = self.dense4(layers.concatenate([x, x3]))
		output = self.dense5(x)
		# x = self.global_pool1([x2, i])
		# x = tf.expand_dims(x, axis=-1)
		# x = self.conv1d1(x)


		return output


class RobustMAE():
	# def __init__(self,
	# 		   reduction=losses_utils.ReductionV2.AUTO,
	# 		   name='mean_absolute_error'):
	# 	super(RobustMAE, self).__init__(
	# 	self.mean_absolute_error, name=name, reduction=reduction)

	def mean_absolute_error(self, y_true, y_pred):
		mean, sigma = tf.split(y_pred, 2, axis=-1)
		# mae = K.abs(mean - y_true)
		mae = K.abs(scaler.denorm(mean) - scaler.denorm(y_true))
		# print(K.mean(mae), self.tensor_scaler.denorm(K.mean(mae)))
		return K.mean(mae)




if __name__=='__main__':
	# train_path='data/databases/MP_formation_energy/train_with_Z.csv'
	# val_path = 'data/databases/MP_formation_energy/test_with_Z.csv'
	# train_path='data/databases/OQMD_Formation_Enthalpy/train.csv'
	# val_path = 'data/databases/OQMD_Formation_Enthalpy/test.csv'
	train_path='data/databases/MP/energy_per_atom/train.csv'
	val_path = 'data/databases/MP/energy_per_atom/val.csv'
	dataset_tr = DataLoader(train_path, is_train=True)
	dataset_val = DataLoader(val_path, is_train=False)
	# dataset_tr = DataLoader('data/databases/mp_bulk_modulus/train.csv')
	# dataset_val = DataLoader('data/databases/mp_bulk_modulus/test.csv')
	# dataset_te = DataLoader('data/databases/MP_formation_energy/test.csv')
	# loader.read()
	# print(dataset[0])
	learning_rate = 3e-4  # Learning rate
	epochs = 800  # Number of training epochs
	batch_size = 128  # Batch size

		
	# Parameters
	F = dataset_tr.n_node_features  # Dimension of node features
	N = dataset_tr.n_nodes
	S = dataset_tr.n_edge_features  # Dimension of edge features
	n_out = dataset_tr.n_labels  # Dimension of the target
	# print("NUMBER ", F, N, S, n_out)

	# # Train/test split
	# idxs = np.random.permutation(len(dataset))
	# split = int(0.8 * len(dataset))
	# idx_tr, idx_te = np.split(idxs, [split])
	# dataset_tr, dataset_te = dataset[idx_tr], dataset[idx_te]

	loader_tr = DisjointLoader(dataset_tr, batch_size=batch_size, epochs=epochs)
	loader_val = DisjointLoader(dataset_val, batch_size=batch_size, epochs=1)
	# loader_te = DisjointLoader(dataset_te, batch_size=batch_size, epochs=1, shuffle=False)
	
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
	loss_fn = RobustLoss()
	robust_mae = RobustMAE()
	# Y_train = pd.read_csv(train_path)[['target']]
	# Y_val = pd.read_csv(val_path)[['target']]
	# scaler = Normalizer()
	# scaler.fit(np.array(Y_train))
	# Y_train = scaler.norm(Y_train)
	# Y_val = scaler.norm(Y_val)
	# loss_fn = MeanAbsoluteError()
	# huber_loss = Huber()
	# loss_fn = CategoricalCrossentropy()

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
			target=tf.cast(target, tf.float32)
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
			# acc=0
			# loader_val = DisjointLoader(dataset_val, batch_size=batch_size, epochs=1)
			for batch_val in validation_data:
				val_inputs, val_targets = batch_val
				val_predictions = model(val_inputs, training=False)
				# val_loss += loss_fn(val_targets, val_predictions)
				# acc += tf.reduce_mean(categorical_accuracy(val_targets, val_predictions))
				val_loss += robust_mae.mean_absolute_error(val_targets, val_predictions)
				# val_loss_total+=val_loss
			step = 0
			print('\nEpoch: ', epoch_no)
			print("Training Loss: {} ....... Validation loss: {}\n".format(loss / loader_tr.steps_per_epoch, val_loss / loader_val.steps_per_epoch))
			# print('validation accuracy: {}'.format(acc/loader_val.steps_per_epoch))
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
	df_preds = pd.DataFrame({"predictions": preds.flatten()})
	df_preds.to_csv("OQMD__Eform_preds_MPCNN.csv", header=None, index=False)

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