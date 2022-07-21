import numpy as np
import pandas as pd
import math
from scipy.spatial import distance as dist
import scipy.sparse as sp
import json
import warnings
warnings.filterwarnings("ignore")

from utils import Normalizer, NormalizeTensor

from pymatgen.core import Composition, Element, Structure
from matminer.featurizers.composition import ElementFraction

from spektral.data import Dataset, Graph


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
				max_no_atoms=500,
				threshold_radius=4,
				use_crystal_structure=False,
				**kwargs):

		self.data_path = data_path
		self.is_pickle = is_pickle
		self.is_train = is_train
		self.pred_func = pred_func
		self.scaler = scaler
		self.embedding_path = embedding_path
		self.embedding_type = embedding_type
		self.max_no_atoms = max_no_atoms
		self.threshold_radius = threshold_radius
		self.use_crystal_structure = use_crystal_structure
		self.efCls = ElementFraction()
		super().__init__(**kwargs)

	def getEmbeddings(self):
		with open(self.embedding_path+self.embedding_type+'-embedding.json') as file:
			data = json.load(file)
			return dict(data)

	def get_angle_matrix(self, coords1, coords2, units='radians'): # not used
		'''
		get the angle between the vectors connecting each atom to the origin (0, 0, 0)
		'''
		costheta = 1 - dist.cdist(coords1, coords2, 'cosine')
		if units=='radians':
			return np.nan_to_num(np.arccos(costheta))
		else:	# degrees
			return np.nan_to_num(np.degrees(np.arccos(costheta)))

	def gaussian_expansion(self, distance_matrix, dmin=0, dmax=5, step=0.25, std=0.5):
		return np.exp(-(distance_matrix[..., np.newaxis] - np.arange(dmin, dmax, step))**2 / std**2)

	def RBF_expansion(self, distance_matrix, dmin=0, dmax=8, bins=20):#step=0.25 not used
		centers = np.linspace(dmin, dmax, bins)
		lengthscale = np.diff(centers).mean()
		gamma = 1/lengthscale
		return np.exp(-gamma *(distance_matrix[..., np.newaxis] - centers)**2)
	
	def read(self):	
		embeddings = self.getEmbeddings()
		if self.is_pickle:
			df = pd.read_pickle(self.data_path)
		else:
			df = pd.read_csv(self.data_path)

		df['num_atoms'] = df.apply(lambda x: Composition(Composition(x.formula).get_integer_formula_and_factor()[0]).num_atoms, axis=1)
		# discard materials having more than max_no_atoms=500 in the unit cell (for computational efficiency)
		len_df = df.shape[0] # to check if any materials are discarded due to large no of atoms in unit cell
		df = df[(df.num_atoms < self.max_no_atoms)]
		df = df.drop(['num_atoms'], axis=1, errors='ignore')
		df.reset_index(drop=True, inplace=True)
		
		if df.shape[0]!=len_df:
			print('\n===================== WARNING ================================\n')
			# warnings.warn("{0} materials have been discarded from the current dataset because they contain more than {1} atoms in the formula/unit cell".format(len_df-df.shape[0], self.max_no_atoms))
			print("{0} materials have been discarded from the current dataset because they contain more than {1} atoms in the formula/unit cell".format(len_df-df.shape[0], self.max_no_atoms))
			print('\n============================================================\n')

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
			if self.pred_func:	# for dielectric function 
				label = row[2:]	# multi-target 
			else:
				label = row['target']

			
			all_sites = []

			if self.use_crystal_structure:	# crystal graph
				cif_str = row['cif']
				s = Structure.from_str(cif_str, fmt='cif')
				distance_matrix = s.distance_matrix
				angle_matrix = self.get_angle_matrix(s.cart_coords, s.cart_coords)
				elem_dict = s.composition.get_el_amt_dict()
				all_sites = []
				for e, n in elem_dict.items():
					el = Element(e)
					all_sites.extend([[el, n]]*int(n))

				
				A = ((distance_matrix<self.threshold_radius)).astype(int) # adjacency matrix
				np.fill_diagonal(A, 0)

				x, y = np.where(A==1)
				expanded_distances = self.gaussian_expansion(distance_matrix)
				E = expanded_distances[x, y] # edge attribute

			else:	# integer formula graph
				comp = Composition(Composition(formula).get_integer_formula_and_factor()[0])
				elem_dict = comp.get_el_amt_dict()
				for e, n in elem_dict.items():
					el = Element(e)
					all_sites.extend([[el, n]]*int(n))
				
				distance_matrix = 1-np.eye((len(all_sites))) 	# all edges have a weight of unity
				A = 1-np.eye((len(all_sites))) # adjacency matrix
				x, y = np.where(A==1)
				E = np.expand_dims(distance_matrix[x, y], axis=-1) # edge attribute

			X = [] # node attribute
			for e in all_sites:	
				X.append(embeddings[e[0].name]) 

			if self.pred_func:
				Y = np.array(label, dtype=np.float)
			else:
				Y = label

			A = sp.csr_matrix(A) # spektral's adjacency matrix should be sparse
			X_list.append(np.array(X))
			A_list.append(A)
			E_list.append(E)
			Y_list.append(Y)

		df.apply(make_graph, axis=1)	# runs the loop
		graph_list = [Graph(x=x, a=a, e=e, y=y) for x, a, e, y in zip(X_list, A_list, E_list, Y_list)]
		return graph_list
















