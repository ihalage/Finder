import numpy as np
import pandas as pd
import math
from scipy.spatial import distance as dist
import scipy.sparse as sp
import json
import warnings
warnings.filterwarnings("ignore")

from utils import ElemProp
from utils import Normalizer, NormalizeTensor

from pymatgen import Composition, Element, Structure
from matminer.featurizers.composition import ElementFraction

from spektral.data import Dataset, Graph


# scaler = Normalizer()

class DataLoader(Dataset):
	"""
	A sub-class to load custom dataset
	"""
	def __init__(self, 
				data_path,
				scaler,
				is_pickle=False,
				is_train=True,
				pred_func=False,
				embedding_path='data/embeddings/',
				embedding_type='mat2vec',
				task_type='regression',
				max_no_atoms=500,
				predict_dist_mat=False,
				use_edge_predictor=False,
				use_crystal_structure=False,
				**kwargs):

		self.data_path = data_path
		self.is_pickle = is_pickle
		self.is_train = is_train
		self.pred_func = pred_func
		self.scaler = scaler#Normalizer()
		self.embedding_path = embedding_path
		self.embedding_type = embedding_type
		self.task_type = task_type
		self.max_no_atoms = max_no_atoms
		self.predict_dist_mat = predict_dist_mat
		# self.use_scaler = use_scaler
		self.use_edge_predictor = use_edge_predictor
		self.use_crystal_structure = use_crystal_structure
		self.ElemProp = ElemProp()
		self.efCls = ElementFraction()
		# self.znet = tf.keras.models.load_model('saved_models/best_model_cnn_Z.h5')
		# self.dismatnet =  tf.keras.models.load_model('saved_models/best_model_cnn_DisMat.h5')
		# self.soapnet = tf.keras.models.load_model('saved_models/best_model_cnn_SoapNet.h5')
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
		df_a = df.apply(lambda x: self.efCls.featurize(Composition(x.formula)), axis=1, result_type='expand')
		df_featurized = pd.concat([df, df_a], axis='columns')
		X = df_featurized.drop(['ID', 'formula', 'integer_formula', 'nsites', 'Z', 'target', 'cif', 'nelements', 'is_inert_gas'], axis=1, errors='ignore')
		return np.expand_dims(X, axis=-1)

	def get_angle_matrix(self, coords1, coords2, units='radians'):
		'''
		get the angle between the vectors connecting each atom to the origin (0, 0, 0)
		'''
		costheta = 1 - dist.cdist(coords1, coords2, 'cosine')
		if units=='radians':
			return np.nan_to_num(np.arccos(costheta))
		else:	# degrees
			return np.nan_to_num(np.degrees(np.arccos(costheta)))

	def gaussian_expansion(self, distance_matrix, dmin=0, dmax=5, step=0.25, std=0.5):#step=0.25
		return np.exp(-(distance_matrix[..., np.newaxis] - np.arange(dmin, dmax, step))**2 / std**2)

	def RBF_expansion(self, distance_matrix, dmin=0, dmax=8, bins=20):#step=0.25
		centers = np.linspace(dmin, dmax, bins)
		lengthscale = np.diff(centers).mean()
		gamma = 1/lengthscale
		# dismat = np.expand_dims(distance_matrix, axis=1)
		# return np.exp(-gamma * (dismat - centers) ** 2)
		return np.exp(-gamma *(distance_matrix[..., np.newaxis] - centers)**2)

		
	def read(self):	# n_nodes = n_element type
		embeddings = self.getEmbeddings()
		if self.is_pickle:
			df = pd.read_pickle(self.data_path)
		else:
			df = pd.read_csv(self.data_path)#.iloc[:1000]

		def analyse_comp(row):
			nelements = len(Composition(row.formula).elements)
			has_inert_gas = any(i in Composition(row.formula).elements for i in [Element('He'), Element('Ne'), Element('Ar'), Element('Kr'), Element('Xe')])
			num_atoms = Composition(Composition(row.formula).get_integer_formula_and_factor()[0]).num_atoms
			return nelements, has_inert_gas, num_atoms

		df['nelements'], df['has_inert_gas'], df['num_atoms'] = zip(*df.apply(analyse_comp, axis=1))
		df = df[(df.nelements>1) & (df.has_inert_gas==False) & (df.num_atoms < self.max_no_atoms)]
		df = df.drop(['nelements', 'has_inert_gas', 'num_atoms'], axis=1, errors='ignore')
		# graph_list = []	# final list of graphs to be returned
		df.reset_index(drop=True, inplace=True)

		if self.predict_dist_mat:
			dismatnet =  tf.keras.models.load_model('saved_models/best_model_cnn_DisMat.h5')
			X = self.parse_formulae(data=df)
			dismat_list = np.squeeze(dismatnet.predict([X]))

		if not self.pred_func:
			if self.is_train:
				self.scaler.fit(np.array(df['target']))
			df['target'] = self.scaler.norm(np.array(df['target']))

		# loop of function calls to create a list of graphs
		X_list = []
		A_list = []
		E_list = []
		Y_list = []
		def make_graph(row):
			idx = row.name
			formula = row['formula']
			if self.pred_func:
				# print(row)
				label = row[2:]
				# print(len(label))
			else:
				label = row['target']

			comp = Composition(Composition(formula).get_integer_formula_and_factor()[0])
			elem_dict = comp.get_el_amt_dict()
			all_sites = []
			for e, n in elem_dict.items():
				el = Element(e)
				# all_sites.extend([el]*int(n))
				all_sites.extend([[el, n]]*int(n))


			if self.use_crystal_structure:
				cif_str = row['cif']
				s = Structure.from_str(cif_str, fmt='cif')
				distance_matrix = s.distance_matrix
				angle_matrix = self.get_angle_matrix(s.cart_coords, s.cart_coords)
				# overwrite all_sites if crystal structure is used. Z can be >1
				elem_dict = s.composition.get_el_amt_dict()
				all_sites = []
				for e, n in elem_dict.items():
					el = Element(e)
					# all_sites.extend([el]*int(n))
					all_sites.extend([[el, n]]*int(n))
			## if we want to use available crystal structures in training and predict edges
			elif self.use_edge_predictor and self.is_train:
				cif_str = row['cif']
				s = Structure.from_str(cif_str, fmt='cif')
				total_atoms = Composition(s.formula).num_atoms
				int_form_atoms = Composition(formula).num_atoms
				z = int(total_atoms/int_form_atoms)

				comp_dict = Composition(s.formula).get_el_amt_dict()
				sites = [s[l] for l in range(s.num_sites)]
				norms=np.linalg.norm(s.distance_matrix, ord=2, axis=1)
				j = 0
				sorted_ids = []
				for e,n in comp_dict.items():
					# print(e, n)
					sorted_ids.extend(list((np.argsort(norms[j:j+int(n)])+j)[:int(n/z)]))
					j=j+int(n)
				for k in range(int(s.num_sites/z)):
					s[k] = sites[sorted_ids[k]]
				s.remove_sites(range(int(s.num_sites/z), s.num_sites))
				distance_matrix = s.distance_matrix
			else:
				distance_matrix = 1-np.eye((len(all_sites))) 	# else all edges have a weight of unity //revert
				# distance_matrix = np.eye((len(all_sites)))
			# print(distance_matrix)
			N = len(all_sites)
			## get adjacency matrix (A). Atoms are densely connected
			# A = 1-np.eye((len(all_sites)))
			if self.use_crystal_structure:
				# A = 1-np.eye((len(all_sites)))
				A = ((distance_matrix<4)).astype(int)
				# np.fill_diagonal(distance_matrix, 100)	# dummy fill value
				# min_distances = np.amin(distance_matrix, axis=1)
				# min_dist_bonds = (distance_matrix==min_distances[:,None])
				# A = np.logical_or(A, min_dist_bonds).astype(int)
				np.fill_diagonal(A, 0)
				
				# print(formula)
				# print(A,'\n\n')
				# all_zeros = not np.any(A)
				# if all_zeros==True:
				# 	print('All zeros in A with bond length=4')
				# 	print(formula,'\n\n')
			else:
				A = 1-np.eye((len(all_sites)))
			# A = np.eye((len(all_sites)))

			## get node attributes
			X = []
			i=0
			for e in all_sites:	#/rev		
				X.append(embeddings[e[0].name]) #//try adding atomic num +[i/100.0, e[1]/100.0]	# normalize by a factor for better performance
				i+=1
				if i==e[1]:
					i=0

			x, y = np.where(A==1)
			if self.use_crystal_structure:
				# distances = distance_matrix[x, y]
				# angles = angle_matrix[x, y]
				# E = np.column_stack((distances, angles))
				
				
				expanded_distances = self.gaussian_expansion(distance_matrix)
				# expanded_distances = self.RBF_expansion(distance_matrix)
				E = expanded_distances[x, y]
				
				# expanded_distances = self.gaussian_expansion(distance_matrix)
				# expanded_angles = self.gaussian_expansion(angle_matrix, dmin=0, dmax=2.5, step=0.125, std=0.1)
				# E = np.column_stack((expanded_distances[x, y], expanded_angles[x, y]))
				# print(E.shape)
				# E = np.expand_dims(distance_matrix[x, y], axis=-1)
			else:
				E = np.expand_dims(distance_matrix[x, y], axis=-1)
			
			if self.pred_func:
				Y = np.array(label, dtype=np.float)
			else:
				Y = label

			A = sp.csr_matrix(A)
			X_list.append(np.array(X))
			A_list.append(A)
			E_list.append(E)
			Y_list.append(Y)

		df.apply(make_graph, axis=1)	# runs the loop
		graph_list = [Graph(x=x, a=a, e=e, y=y) for x, a, e, y in zip(X_list, A_list, E_list, Y_list)]
		return graph_list
















