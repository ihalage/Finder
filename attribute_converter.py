import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pymatgen import Composition
# plt.rcParams['image.cmap'] = 'viridis'
plt.rcParams['image.cmap'] = 'cividis'

'''
# edge = np.array([[0.0540234037],
#  [0.0540234372],
#  [-0.0638993829],
#  [-0.0638993829],
#  [-0.063899368],
#  [-0.274984658],
#  [0.15007484],
#  [-0.0701581389],
#  [-0.0886937678],
#  [-0.0886937231],
#  [-0.0886937231],
#  [-0.267391741],
#  [0.15007484],
#  [-0.0701581836],
#  [-0.0886937678],
#  [-0.0886937082],
#  [-0.0886937231],
#  [-0.267391741],
#  [0.262818217],
#  [0.149473846],
#  [0.149473876],
#  [0.0567170605],
#  [0.0567170307],
#  [-0.171995327],
#  [0.262818217],
#  [0.149473861],
#  [0.149473876],
#  [0.0567170382],
#  [0.0567170307],
#  [-0.171995327],
#  [0.262818187],
#  [0.149473876],
#  [0.149473876],
#  [0.0567170233],
#  [0.0567170419],
#  [-0.171995327],
#  [0.254195333],
#  [0.182999223],
#  [0.182999253],
#  [-0.000865004957],
#  [-0.000865004957],
#  [-0.000864997506]])

fig, ax = plt.subplots(1,4, figsize=(16,8))
edge = np.array([[-0.283914506],
 [-0.351525545],
 [-0.351525545],
 [-0.351525545],
 [-0.446971357],
 [-0.324827135],
 [-0.324827135],
 [-0.324827135],
 [-0.300388753],
 [-0.178154975],
 [-0.212954804],
 [-0.212954715],
 [-0.300388753],
 [-0.178154975],
 [-0.212954745],
 [-0.212954715],
 [-0.300388753],
 [-0.17815499],
 [-0.212954715],
 [-0.212954774]])


edge = edge.flatten()

dismat = np.zeros((int(np.sqrt(len(edge)))+1, int(np.sqrt(len(edge)))+1))

indices = np.where(~np.eye(dismat.shape[0],dtype=bool))
# ind_upper = np.triu_indices(int(np.sqrt(len(edge)))+1, 1)
# ind_lower = np.tril_indices(int(np.sqrt(len(edge)))+1, -1)

# dismat[ind_upper[0], ind_upper[1]] = np.split(edge,2)[0]
# dismat[ind_lower[0], ind_lower[1]] = np.split(edge,2)[1]

dismat[indices[0], indices[1]] = edge
dismat = dismat#+3.8

df = pd.DataFrame(dismat)


print(df)
im = ax[0].imshow(dismat, vmin=-0.6, vmax=0.3)


edge = np.array([[-0.163011014],
 [-0.291645586],
 [-0.291645586],
 [-0.291645586],
 [-0.238533214],
 [-0.248632029],
 [-0.248632],
 [-0.248631969],
 [-0.141145706],
 [-0.0591735095],
 [-0.204129592],
 [-0.204129592],
 [-0.141145706],
 [-0.0591734946],
 [-0.204129592],
 [-0.204129592],
 [-0.141145706],
 [-0.0591734797],
 [-0.204129621],
 [-0.204129592]]
)


edge = edge.flatten()

dismat = np.zeros((int(np.sqrt(len(edge)))+1, int(np.sqrt(len(edge)))+1))

indices = np.where(~np.eye(dismat.shape[0],dtype=bool))
# ind_upper = np.triu_indices(int(np.sqrt(len(edge)))+1, 1)
# ind_lower = np.tril_indices(int(np.sqrt(len(edge)))+1, -1)

# dismat[ind_upper[0], ind_upper[1]] = np.split(edge,2)[0]
# dismat[ind_lower[0], ind_lower[1]] = np.split(edge,2)[1]

dismat[indices[0], indices[1]] = edge
dismat = dismat#+3.8

df = pd.DataFrame(dismat)


print(df)
im = ax[1].imshow(dismat, vmin=-0.6, vmax=0.3)





edge = np.array([[-0.193895116],
 [-0.320945442],
 [-0.320945442],
 [-0.320945442],
 [-0.296848297],
 [-0.232279316],
 [-0.232279256],
 [-0.232279316],
 [-0.188247576],
 [-0.0435209721],
 [-0.175568745],
 [-0.175568759],
 [-0.188247576],
 [-0.0435209125],
 [-0.175568745],
 [-0.175568759],
 [-0.188247606],
 [-0.0435209498],
 [-0.175568745],
 [-0.175568745]]
)


edge = edge.flatten()

dismat = np.zeros((int(np.sqrt(len(edge)))+1, int(np.sqrt(len(edge)))+1))

indices = np.where(~np.eye(dismat.shape[0],dtype=bool))
# ind_upper = np.triu_indices(int(np.sqrt(len(edge)))+1, 1)
# ind_lower = np.tril_indices(int(np.sqrt(len(edge)))+1, -1)

# dismat[ind_upper[0], ind_upper[1]] = np.split(edge,2)[0]
# dismat[ind_lower[0], ind_lower[1]] = np.split(edge,2)[1]

dismat[indices[0], indices[1]] = edge
dismat = dismat#+3.8

df = pd.DataFrame(dismat)


print(df)
im = ax[2].imshow(dismat, vmin=-0.6, vmax=0.3)


edge = np.array([[0.010497544],
 [-0.503124893],
 [-0.503124952],
 [-0.503124893],
 [-0.350290775],
 [-0.541505694],
 [-0.541505754],
 [-0.541505694],
 [-0.312286735],
 [0.201524377],
 [-0.380431354],
 [-0.380431354],
 [-0.312286735],
 [0.201524362],
 [-0.380431354],
 [-0.380431354],
 [-0.312286735],
 [0.201524362],
 [-0.380431414],
 [-0.380431354]]

)


edge = edge.flatten()

dismat = np.zeros((int(np.sqrt(len(edge)))+1, int(np.sqrt(len(edge)))+1))

indices = np.where(~np.eye(dismat.shape[0],dtype=bool))
# ind_upper = np.triu_indices(int(np.sqrt(len(edge)))+1, 1)
# ind_lower = np.tril_indices(int(np.sqrt(len(edge)))+1, -1)

# dismat[ind_upper[0], ind_upper[1]] = np.split(edge,2)[0]
# dismat[ind_lower[0], ind_lower[1]] = np.split(edge,2)[1]

dismat[indices[0], indices[1]] = edge
dismat = dismat#+3.8

df = pd.DataFrame(dismat)


print(df)
im = ax[3].imshow(dismat, vmin=-0.6, vmax=0.3)



fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

plt.show()



f = open('node_attributes_BaTiO3', 'r')
txt = f.readlines()
a = []
for line in txt:
	l = line.replace(" ", ",").replace('\n', '')
	# print(l)
	a.append(l)

c=''.join(a)
arr = np.array(eval(c))
plt.imshow(arr, aspect=20)
plt.show()
print(arr.shape)
# a = np.loadtxt('node_attributes')
# print(a.shape)
'''

fig, ax = plt.subplots(2, 4, figsize=(10,4))

materials = ['SrTiO3', 'BaTiO3', 'KNbO3', 'CsPbI3']
properties = ['Ef', 'refractive']

row = 0
col = 0
for p in properties:
	for m in materials:

		comp = Composition(m)
		elements = []
		for k,v in comp.get_el_amt_dict().items():
			elements.extend([k]*int(v))
		# elements = np.array(elements)
		filename = 'edges_and_nodes/edges/edge_attribute_{0}_{1}'.format(p, m)
		f = open(filename, 'r')
		c = f.readlines()
		attribute_list = []

		for line in c:
			attribute_list.append(line.replace('\n', ','))

		attribute_list[-1]=attribute_list[-1][:-1]
		attribute_list = ''.join(attribute_list)

		attribute_arr = np.array(eval(attribute_list))
		# print(attribute_arr.shape)

		edge = attribute_arr.flatten()
		dismat = np.zeros((int(np.sqrt(len(edge)))+1, int(np.sqrt(len(edge)))+1))
		indices = np.where(~np.eye(dismat.shape[0],dtype=bool))

		dismat[indices[0], indices[1]] = edge
		dismat = dismat
		df = pd.DataFrame(dismat)

		print('\n', p, m)
		print(df)
		
		ax[row, col].set_xticks(range(len(elements)))
		ax[row, col].set_xticklabels(elements, rotation='horizontal', fontsize=10)

		ax[row, col].set_yticks(range(len(elements)))
		ax[row, col].set_yticklabels(elements,  fontsize=10)
		# ax[row, col].set_aspect('equal')


		# ax[row, col].set_xticks(range(len(elements)),elements)
		# ax[row, col].set_yticks(range(len(elements)), elements)
		im = ax[row, col].imshow(dismat, aspect=0.8, vmin=-0.8, vmax=1)
		
		# plt.imshow(dismat, vmin=-0.6, vmax=0.9)
		# plt.show()
		# im = ax[1].imshow(dismat, vmin=-0.6, vmax=0.3)



		# filename = 'edges_and_nodes/nodes/node_attributes_{0}_{1}'.format(p, m)
		# f = open(filename, 'r')
		# c = f.readlines()
		# nodes = []

		# for line in c:
		# 	l = line.replace(" ", ",").replace('\n', '')
		# 	nodes.append(l)

		# nodes=''.join(nodes)
		# node_arr = np.array(eval(nodes))
		# ax[row+1, col].set_yticks(range(len(elements)))
		# ax[row+1, col].set_yticklabels(elements,  fontsize=8)
		# ax[row+1, col].imshow(node_arr, aspect=12, vmin=-0.2, vmax=0.3)
		# ax[row+1, col].set_xticks([])
		# # plt.imshow(node_arr, aspect=10)
		# # plt.show()

		col+=1
	row+=1
	col = 0
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax)
# fig.colorbar(im, pad=0.25)
fig.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=0.89, top=None, wspace=0.4, hspace=0.1)

# fig.tight_layout()
fig.savefig('edge_matrix.pdf', dpi=200)
plt.show()

