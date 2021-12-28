'''
This script creates SOAP embeddings for the crystal structures available in the materials project database

@Achintha_Ihalage
Jul 2021
'''

import numpy as np
import pandas as pd
from dscribe.descriptors import SOAP

import ase.io as io
from ase.spacegroup import get_spacegroup
from io import StringIO

import scipy.sparse as sp
import pickle


species = range(1, 104)
rcut = 6.0
nmax = 2
lmax = 2

# Setting up the local SOAP descriptor
soap_local = SOAP(
    species=species,
    periodic=True,
    rcut=rcut,
    nmax=nmax,
    lmax=lmax,
    sparse=True,
)

# Setting up global soap descriptor
# Setting up the SOAP descriptor
soap_global = SOAP(
    species=species,
    periodic=True,
    rcut=rcut,
    nmax=nmax,
    lmax=lmax,
    average="inner",
    sparse=True
)


def make_soap(data_path, soap_type='global', task='train'):
	df = pd.read_csv(data_path)#.iloc[:100]
	if soap_type=="local":
		soap = soap_local
	else:
		soap = soap_global
	# cif = df.iloc[0]['cif']
	soap_csr_list = []
	for index, row in df.iterrows():
		f = StringIO(row.cif)
		structure = io.read(f, format='cif')
		soap_cif = soap.create(structure)
		print(index)
		soap_csr_list.append(soap_cif)
	# a = np.array(soap_csr_list, dtype='object')
	# a = sp.csr_matrix(soap_csr_list)
	# print(a.nbytes)
	with open('data/databases/MP_formation_energy/soap_{0}_{1}.pkl'.format(soap_type, task), 'wb') as f:
		pickle.dump(soap_csr_list, f)
		# data = pickle.rea
	# data = pd.read_pickle('parrot.pkl')
	# print(data[0].todense())

	# asp = sp.csr_matrix(a)
	# print(asp)
	# print(np.array(soap_csr_list))
	# soap_csr = sp.vstack(soap_csr_list)
	# print(soap_csr.shape)

make_soap('data/databases/MP_formation_energy/train.csv')
make_soap('data/databases/MP_formation_energy/test.csv', task='test')

