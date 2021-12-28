import numpy as np
import pandas as pd
import cv2
from scipy import sparse
import pickle

from pymatgen import Element, Structure, Composition
from matminer.featurizers.composition import ElementFraction

import matplotlib.pyplot as plt
from matplotlib import rcParams

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
# from keras_self_attention import SeqSelfAttention
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf

import joblib
from sklearn.datasets import load_linnerud
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.39
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


# df_train = pd.read_csv('data/databases/MP_formation_energy/train_with_Z.csv')#.iloc[:1000]
# df_test = pd.read_csv('data/databases/MP_formation_energy/test_with_Z.csv')#.iloc[:1000]

# df_train['nelements'] = df_train['formula'].apply(lambda x: len(Composition(x).elements))
# df_test['nelements'] = df_test['formula'].apply(lambda x: len(Composition(x).elements))

# df_train['is_inert_gas'] = df_train['formula'].apply(lambda x: any(i in Composition(x).elements for i in [Element('He'), Element('Ne'), Element('Ar'), Element('Kr'), Element('Xe')]))
# df_test['is_inert_gas'] = df_test['formula'].apply(lambda x: any(i in Composition(x).elements for i in [Element('He'), Element('Ne'), Element('Ar'), Element('Kr'), Element('Xe')]))

# df_train['num_atoms'] = df_train['formula'].apply(lambda x: Composition(Composition(x).get_integer_formula_and_factor()[0]).num_atoms)
# df_test['num_atoms'] = df_test['formula'].apply(lambda x: Composition(Composition(x).get_integer_formula_and_factor()[0]).num_atoms)

# df_train = df_train[(df_train.nelements>1) & (df_train.is_inert_gas==False) & (df_train.num_atoms <=40)]
# df_test = df_test[(df_test.nelements>1) & (df_test.is_inert_gas==False) & (df_test.num_atoms <=40)]

# df_train = df_train[df_train.num_atoms<=40]
# df_test = df_test[df_test.num_atoms<=40]

# dist_mat_list_tr = []
# dist_mat_list_te = []

# for i, row in df_train.iterrows():
# 	dm_resized = np.zeros((40, 40))
# 	# if row.Z<=32:
# 	z=row.Z
# 	cif = row['cif']
# 	# fig, ax = plt.subplots(1,2)
# 	s = Structure.from_str(cif, fmt='cif')
# 	comp_dict = Composition(s.formula).get_el_amt_dict()
# 	full_formula = s.formula
# 	# ax[0].imshow(s.distance_matrix)
# 	# dm = s.distance_matrix
# 	sites = [s[l] for l in range(s.num_sites)]
# 	norms=np.linalg.norm(s.fractional_distance_matrix, ord=2, axis=1)
# 	j = 0
# 	sorted_ids = []
# 	for e,n in comp_dict.items():
# 		# print(e, n)
# 		sorted_ids.extend(list((np.argsort(norms[j:j+int(n)])+j)[:int(n/z)]))
# 		j=j+int(n)
# 	for k in range(int(s.num_sites/z)):
# 		s[k] = sites[sorted_ids[k]]
# 	s.remove_sites(range(int(s.num_sites/z), s.num_sites))
# 	dm = s.fractional_distance_matrix
# 	dm_resized[:dm.shape[0],:dm.shape[1]] = dm
# 	# dm_resized = sparse.csr_matrix(dm_resized)
# 	# ax[1].imshow(dm)
# 	# plt.show()
# 	# print(full_formula, Composition(s.formula).get_integer_formula_and_factor()[0], dm.shape)
# 	# dm_resized = cv2.resize(dm, dsize=(10, 10))
# 	dist_mat_list_tr.append(dm_resized)

# for i, row in df_test.iterrows():
# 	dm_resized = np.zeros((40, 40))
# 	# if row.Z<=32:
# 	z=row.Z
# 	cif = row['cif']
# 	# fig, ax = plt.subplots(1,2)
# 	s = Structure.from_str(cif, fmt='cif')
# 	comp_dict = Composition(s.formula).get_el_amt_dict()
# 	full_formula = s.formula
# 	# ax[0].imshow(s.distance_matrix)
# 	# dm = s.distance_matrix
# 	sites = [s[l] for l in range(s.num_sites)]
# 	norms=np.linalg.norm(s.fractional_distance_matrix, ord=2, axis=1)
# 	j = 0
# 	sorted_ids = []
# 	for e,n in comp_dict.items():
# 		# print(e, n)
# 		sorted_ids.extend(list((np.argsort(norms[j:j+int(n)])+j)[:int(n/z)]))
# 		j=j+int(n)
# 	for k in range(int(s.num_sites/z)):
# 		s[k] = sites[sorted_ids[k]]
# 	s.remove_sites(range(int(s.num_sites/z), s.num_sites))
# 	dm = s.fractional_distance_matrix
# 	dm_resized[:dm.shape[0],:dm.shape[1]] = dm
# 	# dm_resized = sparse.csr_matrix(dm_resized)
# 	# ax[1].imshow(dm)
# 	# plt.show()
# 	# print(full_formula, Composition(s.formula).get_integer_formula_and_factor()[0], dm.shape)
# 	# dm_resized = cv2.resize(dm, dsize=(10, 10))
# 	dist_mat_list_te.append(dm_resized)

# dist_mat_tr = np.array(dist_mat_list_tr)
# dist_mat_te = np.array(dist_mat_list_te)

# print(dist_mat_tr.shape, dist_mat_te.shape)

# np.save('data/databases/MP_formation_energy/dist_mat_tr_fractional_dis_mat_40x40_padded_gnn.npy', dist_mat_tr)
# np.save('data/databases/MP_formation_energy/dist_mat_te_fractional_dis_mat_40x40_padded_gnn.npy', dist_mat_te)

# with open('data/databases/MP_formation_energy/dis_mat_tr_500x500.pkl', 'wb') as f:
# 	pickle.dump(dist_mat_list_tr, f)

# with open('data/databases/MP_formation_energy/dis_mat_te_500x500.pkl', 'wb') as f:
# 	pickle.dump(dist_mat_list_te, f)




# def get_attributes(row):
# 	cif = row['cif']
# 	s = Structure.from_str(cif, fmt='cif')
# 	nsites = s.num_sites
# 	int_comp = Composition(Composition(s.formula).get_integer_formula_and_factor()[0])
# 	num_atoms_int_form = int_comp.num_atoms
# 	Z = nsites/num_atoms_int_form
	
# 	dm = np.pad(dm, [(0, 450-dm.shape[0]), (0, 450-dm.shape[0])], mode='constant')
# 	dist_mat_list.append(dm)
# 	return int_comp.formula, Z, nsites
# 	return dm_resized


# df_train['integer_formula'], df_train['Z'], df_train['nsites'] = zip(*df_train.apply(get_attributes, axis=1))
# df_test['integer_formula'], df_test['Z'], df_test['nsites'] = zip(*df_test.apply(get_attributes, axis=1))

# df_train['distance_matrix'] = df_train.apply(get_attributes, axis=1)
# df_test['distance_matrix'] = df_test.apply(get_attributes, axis=1)



# print(df_train['nsites'].max())
# print(df_train['Z'].max())

# print(df_test['nsites'].max())
# print(df_test['Z'].max())


# df_train = df_train[df_train.Z<=32]

# print(df_train)



# dist_mat_list = np.array(dist_mat_list)
# print(dist_mat_list.shape)

# df_train.to_csv('train_with_Z.csv', index=False)
# df_test.to_csv('test_with_Z.csv', index=False)





class ZNet(object):
	def __init__(self, 
				total_elements=103,
				n_classes=32,
				):
		self.total_elements = total_elements
		self.n_classes = n_classes
		self.efCls = ElementFraction()

	def build_ZNet(self, input_shape):
		# input_shape = (self.X_train.shape[1], 1)

		inputs = tf.keras.Input(shape=input_shape)

		x = layers.Conv1D(64,3, activation='relu')(inputs)
		x = layers.Conv1D(128,3, activation='relu')(x)
		x = layers.Conv1D(256,3, activation='relu')(x)
		# x = layers.Conv1D(256,3, activation='relu')(x)
		# x = layers.Conv1D(256,3, activation='relu')(x)
		x = layers.GlobalMaxPooling1D()(x)

		x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x)
		x = layers.Dropout(0.1)(x)

		x = layers.Dense(1024, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x)
		x = layers.Dropout(0.1)(x)

		# x = layers.Concatenate()([x, layers.Flatten()(x3)])

		x = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x)
		x = layers.Dropout(0.1)(x)

		output = layers.Dense(self.n_classes, activation='softmax')(x)
		model = tf.keras.Model([inputs], [output], name="ZNet")
		return model


	def parse_formulae(self, data_path=None, data=None):
		if data_path is not None:
			df = pd.read_csv(data_path, keep_default_na = False)
		if data is not None:
			df = data
		df = df[df.Z <= self.n_classes]
		print(df.info())
		df_a = df.apply(lambda x: self.efCls.featurize(Composition(x.formula)), axis=1, result_type='expand')
		df_featurized = pd.concat([df, df_a],axis='columns')
		X = df_featurized.drop(['ID', 'formula', 'integer_formula', 'Z', 'nsites', 'target', 'cif'], axis=1, errors='ignore')
		n_samples = df.shape[0]
		Y = np.zeros((n_samples, self.n_classes))
		y_cls = np.array(df_featurized['Z']).astype(int)
		Y[range(n_samples), y_cls-1] = 1

		# X_frac = self.generate_Z_frac(X)

		return np.expand_dims(X, axis=-1).astype('float32'), np.array(Y, dtype=np.float)

	def train_ZNet(self, train_path=None, val_path=None, data=None, lr=0.0001, batch_size=64, epochs=800, random_state=10):
		print("Loading and processing data ...")
		if (train_path and val_path):
			X_train, y_train = self.parse_formulae(data_path=train_path)
			X_val, y_val = self.parse_formulae(data_path=val_path)

		elif data is not None:
			X_data, y_data = self.parse_formulae(data=data)
			X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.15, random_state=random_state)
			# X_train_frac, X_val_frac, _, _ = train_test_split(X_data_frac, y_data, test_size=0.15, random_state=random_state)

		else:
			raise ValueError("Data path or a dataframe containing the data should be provided")

		input_shape = (X_train.shape[1], X_train.shape[2])

		# Z_frac_shape = (X_train_frac.shape[1], X_train_frac.shape[2])

		model_cnn = self.build_ZNet(input_shape)
		# model_dense = self.build_model(with_cnn=False)
		print(model_cnn.summary())
		model_cnn.compile(
					loss='categorical_crossentropy',
					optimizer=tf.keras.optimizers.Adam(learning_rate=lr), #tfa.optimizers.LAMB(learning_rate=lr),
					metrics=['accuracy'])

		

		checkpointer_cnn = tf.keras.callbacks.ModelCheckpoint('saved_models/best_model_cnn_Z.h5', monitor='accuracy', save_best_only=True)#attn_conv was best model 0.0294
		# checkpointer_dense = tf.keras.callbacks.ModelCheckpoint('saved_models/Rex_best_model_dense_clipped.h5', save_best_only=True)
		try:
			# history_cnn = model_cnn.fit([np.expand_dims(np.array(self.X_train), axis=-1), np.expand_dims(np.array(self.X_train_frac), axis=-1)], self.y_train, validation_split=0.15, epochs=epochs, batch_size=batch_size, callbacks=[checkpointer_cnn])
			# history_cnn = model_cnn.fit([np.expand_dims(np.array(self.X_train_wavelet), axis=-1)], self.y_train, validation_split= 0.15, epochs=epochs, batch_size=batch_size, callbacks=[checkpointer_cnn])			
			# history_cnn = model_cnn.fit(np.expand_dims(np.array(self.X_train)[:10000], axis=-1), self.y_train[:10000], validation_data=(np.expand_dims(np.array(self.X_test), axis=-1), self.y_test), epochs=epochs, batch_size=batch_size, callbacks=[checkpointer_cnn])
			history_cnn = model_cnn.fit([X_train], y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, callbacks=[checkpointer_cnn])
		except:
			model = tf.keras.models.load_model('saved_models/best_model_cnn_Z.h5')
			# evaluation = model.evaluate(X_val, y_val)
			predictions = np.argmax(model.predict(X_val), axis=1)
			targets = np.argmax(y_val, axis=1)
			df = pd.DataFrame({'target': targets, 'prediction': predictions})
			# print(df.iloc[100:])
			# print(evaluation)

	def test_ZNet(self, model_path, test_path, n_test=100):
		model = tf.keras.models.load_model(model_path, compile=False)

		X_test, y_test = self.parse_formulae(data_path=test_path)

		y_pred = model.predict([X_test])
		# print(df_test[['integer_formula', 'nsites']].iloc[20])
		# print(y_pred[20])
		# print(y_test[20])


class DisMatNet(ZNet):
	def __init__(self,
				dis_mat_shape=(10, 10),
				n_classes=32):
		self.dis_mat_shape = dis_mat_shape
		self.n_classes = n_classes
		self.efCls = ElementFraction()
		# super().__init__(**kwargs)
		self.f =  MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie"),
                         cf.ValenceOrbital(props=['avg']), cf.IonProperty(fast=True)])

	def build_DisMatNet(self, input_shape):
		inputs = tf.keras.Input(shape=input_shape)

		x1 = layers.Conv1D(64,3, activation='relu')(inputs)
		x2 = layers.Conv1D(128,3, activation='relu')(x1)
		x = layers.Conv1D(256,3, activation='relu')(x2)
		x = layers.Conv1D(256,3, activation='relu')(x)
		x = layers.GlobalMaxPooling1D()(x)
		# x = layers.Flatten()(x)

		# x = layers.Dense(256, activation='relu')(x)
		# x = layers.Concatenate()([x, layers.Flatten()(x2)])
		# x = layers.Dense(1024, activation='relu')(x)
		# x = layers.Concatenate()([x, layers.Flatten()(x1)])
		# x = layers.Dense(1024, activation='relu')(x)
		# x = layers.Dense(2048, activation='relu')(x)
		# x = layers.Concatenate()([x, layers.Flatten()(x1)])
		# x = layers.Dense(512, activation='relu')(x)
		# output = layers.Dense(780, activation='linear')(x)
		# outs = []
		# for i in range(780):
		# 	# y = layers.Conv1D(10,3, activation='relu')(inputs)
		# 	# y = layers.GlobalMaxPooling1D()(y)
		# 	# y = layers.Dense(256, activation='relu')(inputs)
		# 	# y = layers.Dense(128, activation='relu')(inputs)
		# 	# y = layers.Dense(256, activation='relu')(y)
		# 	y = layers.Dense(64, activation='relu')(inputs)
		# 	y_out = layers.Dense(1)(y)
		# 	outs.append(y_out)
		# print(outs)
		# print(len(outs))
		# output = layers.Concatenate()(outs)

		

		x = layers.Dense(self.dis_mat_shape[0]*self.dis_mat_shape[1], activation='relu')(x)

		x = layers.Reshape((self.dis_mat_shape[0], self.dis_mat_shape[1], 1))(x)

		# x = tf.expand_dims(x, axis=-1)
		x3 = layers.Conv2D(64,3, activation='relu', padding='same')(x)
		x = layers.Conv2D(128,1, activation='relu', padding='same')(x3)
		# x = layers.Conv2D(256,1, activation='relu', padding='same')(x)
		x = layers.Conv2D(256,1, activation='relu', padding='same')(x)
		x = layers.Concatenate()([x, x3])
		# x = layers.Conv2D(256,3, activation='relu', padding='same')(x)
		output = layers.Conv2D(1, 1, activation='linear', padding='same')(x)

		model = tf.keras.Model([inputs], [output], name="DisMatNet")
		return model

	def parse_formulae(self, data_path=None, data=None):
		if data_path is not None:
			df = pd.read_csv(data_path, keep_default_na = False)
		if data is not None:
			df = data
		df['num_atoms'] = df['formula'].apply(lambda x: Composition(Composition(x).get_integer_formula_and_factor()[0]).num_atoms)
		df = df[df.num_atoms <= 40]
		df_a = df.apply(lambda x: self.efCls.featurize(Composition(x.formula)), axis=1, result_type='expand')
		# df_a = df.apply(lambda x: self.f.featurize(Composition(x.formula)), axis=1, result_type='expand').fillna(0)
		df_featurized = pd.concat([df, df_a], axis='columns')
		X = df_featurized.drop(['ID', 'formula', 'integer_formula', 'nsites', 'target', 'Z', 'cif'], axis=1, errors='ignore')

		# return np.expand_dims(X, axis=-1)#.astype('float32')
		return np.array(X)
		
		# n_samples = df.shape[0]
		# Y = np.zeros((n_samples, self.n_classes))
		# y_cls = np.array(df_featurized['Z']).astype(int)
		# Y[range(n_samples), y_cls-1] = 1

	def train_DisMatNet(self, train_path=None, val_path=None, train_y_path=None, val_y_path=None, data=None, lr=0.0001, batch_size=64, epochs=800, random_state=10):
		print("Loading and processing data ...")
		if (train_path and val_path):
			X_train = self.parse_formulae(data_path=train_path)#[:2000]
			# y_train = np.expand_dims(np.load(train_y_path), axis=-1)#/10.0
			
			def get_uptriangle(array):
				ids = np.triu_indices(array.shape[0], k=1)
				return array[ids]
			y_train = np.load(train_y_path)#/10.0
			y_train = np.array(list(map(get_uptriangle,y_train)))#[:2000]
			X_val = self.parse_formulae(data_path=val_path)#[:2000]
			# y_val = np.expand_dims(np.load(val_y_path), axis=-1)#/10.0
			y_val = np.load(val_y_path)#/10.0
			y_val = np.array(list(map(get_uptriangle,y_val)))#[:2000]

		# elif data is not None:
		# 	X_data, y_data = self.parse_formulae(data=data)
		# 	X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.15, random_state=random_state)
		# 	# X_train_frac, X_val_frac, _, _ = train_test_split(X_data_frac, y_data, test_size=0.15, random_state=random_state)

		# else:
		# 	raise ValueError("Data path or a dataframe containing the data should be provided")

		# input_shape = (X_train.shape[1], X_train.shape[2])
		# # input_shape = (X_train.shape[1], )
		# # # Z_frac_shape = (X_train_frac.shape[1], X_train_frac.shape[2])

		# model_cnn = self.build_DisMatNet(input_shape)
		# # model_dense = self.build_model(with_cnn=False)
		# print(model_cnn.summary())
		# model_cnn.compile(
		# 			loss='mae',
		# 			optimizer=tf.keras.optimizers.Adam(learning_rate=lr), #tfa.optimizers.LAMB(learning_rate=lr),
		# 			metrics=['mse', 'mae'])

		### uncomment for ML
		reg = MultiOutputRegressor(RandomForestRegressor(random_state=9, n_estimators=12, max_depth=40, verbose=3))
		# reg = MultiOutputRegressor(RandomForestRegressor(random_state=9, verbose=3))
		reg.fit(X_train.reshape(X_train.shape[0], X_train.shape[1]), y_train)
		# reg.fit(X_train.reshape(X_train.shape[0], X_train.shape[1]), y_train.reshape(y_train.shape[0], y_train.shape[1]*y_train.shape[2]))
		predicted_distances = reg.predict(X_val.reshape(X_val.shape[0], X_val.shape[1]))
		# mae = mean_absolute_error(y_val.reshape(y_val.shape[0], y_val.shape[1]*y_val.shape[2]), predicted_distances)
		mae = mean_absolute_error(y_val, predicted_distances)
		print(mae)
		y_test = y_val
		print(predicted_distances.shape, y_test.shape)
		filename = 'data/databases/MP_formation_energy/random_forest_full_40deapth.joblib.pkl'
		joblib.dump(reg, filename, compress=3)
		for i in range(30):
			pred = predicted_distances[i+20]
			y = y_test[i+20]
			# print(np.c_[pred[:50], y[:50]],'\n\n')
			
			# print(np.c_[pred, y_test],'\n\n')
			l = len(y)
			pred = pred[:l]
			print(pred.shape, y.shape)
			size_X = int(np.sqrt(2*l))+1
			# put it back into a 2D symmetric array
			# size_X = 3
			X = np.zeros((size_X,size_X))
			X[np.triu_indices(X.shape[0], k = 1)] = pred
			pred = X + X.T

			X = np.zeros((size_X,size_X))
			X[np.triu_indices(X.shape[0], k = 1)] = y
			y = X + X.T
			
			fig, ax = plt.subplots(1,2)
			# im=ax[0].imshow(y_test[60+i].reshape(self.dis_mat_shape))
			# im=ax[1].imshow(y_pred[60+i].reshape(self.dis_mat_shape))
			im = ax[0].imshow(y)
			im = ax[1].imshow(pred)
			fig.subplots_adjust(right=0.8)
			cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
			fig.colorbar(im, cax=cbar_ax)
			plt.show()



		# for i in range(10):
		# 	fig, ax = plt.subplots(1,2)
		# 	# im=ax[0].imshow(y_test[60+i].reshape(self.dis_mat_shape))
		# 	# im=ax[1].imshow(y_pred[60+i].reshape(self.dis_mat_shape))
		# 	N = int(X_val[90+i][0])
		# 	im=ax[0].imshow(np.squeeze(y_val[90+i][:N, :N]))
		# 	im=ax[1].imshow(np.squeeze(pred[90+i].reshape(40, 40)[:N, :N]))
		# 	fig.subplots_adjust(right=0.8)
		# 	cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
		# 	fig.colorbar(im, cax=cbar_ax)

		# 	plt.show()

		checkpointer_cnn = tf.keras.callbacks.ModelCheckpoint('saved_models/best_model_cnn_DisMat.h5', monitor='mae', save_best_only=True)#attn_conv was best model 0.0294
		# checkpointer_dense = tf.keras.callbacks.ModelCheckpoint('saved_models/Rex_best_model_dense_clipped.h5', save_best_only=True)
		try:
			# history_cnn = model_cnn.fit([np.expand_dims(np.array(self.X_train), axis=-1), np.expand_dims(np.array(self.X_train_frac), axis=-1)], self.y_train, validation_split=0.15, epochs=epochs, batch_size=batch_size, callbacks=[checkpointer_cnn])
			# history_cnn = model_cnn.fit([np.expand_dims(np.array(self.X_train_wavelet), axis=-1)], self.y_train, validation_split= 0.15, epochs=epochs, batch_size=batch_size, callbacks=[checkpointer_cnn])			
			# history_cnn = model_cnn.fit(np.expand_dims(np.array(self.X_train)[:10000], axis=-1), self.y_train[:10000], validation_data=(np.expand_dims(np.array(self.X_test), axis=-1), self.y_test), epochs=epochs, batch_size=batch_size, callbacks=[checkpointer_cnn])
			history_cnn = model_cnn.fit([X_train], y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, callbacks=[checkpointer_cnn])
		except:
			model = tf.keras.models.load_model('saved_models/best_model_cnn_DisMat.h5')
			# evaluation = model.evaluate(X_val, y_val)
			predictions = np.argmax(model.predict(X_val), axis=1)
			targets = np.argmax(y_val, axis=1)
			df = pd.DataFrame({'target': targets, 'prediction': predictions})

	def test_DisMatNet(self, model_path, test_path, test_y_path, n_test=30):
		# model = tf.keras.models.load_model(model_path)
		model  = joblib.load('data/databases/MP_formation_energy/random_forest_full_40deapth.joblib.pkl') 

		X_test = self.parse_formulae(data_path=test_path)
		y_test = np.load(test_y_path)#/10.0
		# y_pred = model.predict([X_test])
		y_pred = model.predict(X_test)

		def get_uptriangle(array):
			ids = np.triu_indices(array.shape[0], k=1)
			return array[ids]
		y_test = np.array(list(map(get_uptriangle,y_test)))


		for i in range(40):
			pred = y_pred[i+60]
			y = y_test[i+60]
			
			# print(np.c_[pred, y_test],'\n\n')
			l = len(y)
			pred = pred[:l]
			print(np.c_[pred[:50], y[:50]],'\n\n')
			size_X = int(np.sqrt(2*l))+1
			# put it back into a 2D symmetric array
			# size_X = 3
			X = np.zeros((size_X,size_X))
			X[np.triu_indices(X.shape[0], k = 1)] = pred
			pred = X + X.T

			X = np.zeros((size_X,size_X))
			X[np.triu_indices(X.shape[0], k = 1)] = y
			y = X + X.T
			
			fig, ax = plt.subplots(1,2, figsize=(6, 10))
			# im=ax[0].imshow(y_test[60+i].reshape(self.dis_mat_shape))
			# im=ax[1].imshow(y_pred[60+i].reshape(self.dis_mat_shape))
			im = ax[0].imshow(y)
			im = ax[1].imshow(pred)
			fig.subplots_adjust(right=0.8)
			cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
			fig.colorbar(im, cax=cbar_ax)
			plt.show()

		# for i in range(n_test):
		# 	fig, ax = plt.subplots(1,2)
		# 	# im=ax[0].imshow(y_test[60+i].reshape(self.dis_mat_shape))
		# 	# im=ax[1].imshow(y_pred[60+i].reshape(self.dis_mat_shape))
		# 	N = int(X_test[90+i][0])
		# 	im=ax[0].imshow(np.squeeze(y_test[90+i][:N, :N]))
		# 	im=ax[1].imshow(np.squeeze(y_pred[90+i][:N, :N]))
		# 	fig.subplots_adjust(right=0.8)
		# 	cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
		# 	fig.colorbar(im, cax=cbar_ax)

		# 	plt.show()






			

# if __name__=='__main__':
# 	znet = ZNet()
# 	dismatnet = DisMatNet()
# 	# znet.train_ZNet(train_path='data/databases/MP_formation_energy/train_with_Z.csv', val_path='data/databases/MP_formation_energy/test_with_Z.csv', epochs=100)
# 	# znet.test_ZNet(model_path='saved_models/best_model_cnn_Z.h5', test_path='data/databases/MP_formation_energy/test_with_Z.csv')

# 	dismatnet.train_DisMatNet(train_path='data/databases/MP_formation_energy/train_with_Z.csv',
# 							 val_path='data/databases/MP_formation_energy/test_with_Z.csv', 
# 							 train_y_path='data/databases/MP_formation_energy/dist_mat_tr_fractional_dis_mat_40x40_padded.npy',
# 							 val_y_path='data/databases/MP_formation_energy/dist_mat_te_fractional_dis_mat_40x40_padded.npy',
# 							 epochs=300)
# 	# dismatnet.test_DisMatNet(model_path='saved_models/best_model_cnn_DisMat.h5', 
# 	# 						test_path='data/databases/MP_formation_energy/test_with_Z.csv', 
# 	# 						test_y_path='data/databases/MP_formation_energy/dist_mat_te_fractional_dis_mat_40x40_padded.npy')











'''

###### DL MOdel



import numpy as np
import pandas as pd
import cv2
from scipy import sparse
import pickle

from pymatgen import Element, Structure, Composition
from matminer.featurizers.composition import ElementFraction

import matplotlib.pyplot as plt
from matplotlib import rcParams

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
# from keras_self_attention import SeqSelfAttention
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf

import joblib
from sklearn.datasets import load_linnerud
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.39
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


df_train = pd.read_csv('data/databases/MP_formation_energy/train_with_Z.csv')#.iloc[:1000]
df_test = pd.read_csv('data/databases/MP_formation_energy/test_with_Z.csv')#.iloc[:1000]

# df_train['num_atoms'] = df_train['formula'].apply(lambda x: Composition(Composition(x).get_integer_formula_and_factor()[0]).num_atoms)
# df_test['num_atoms'] = df_test['formula'].apply(lambda x: Composition(Composition(x).get_integer_formula_and_factor()[0]).num_atoms)

# df_train = df_train[df_train.num_atoms<=40]
# df_test = df_test[df_test.num_atoms<=40]

# dist_mat_list_tr = []
# dist_mat_list_te = []

# for i, row in df_train.iterrows():
# 	dm_resized = np.zeros((40, 40))
# 	# if row.Z<=32:
# 	z=row.Z
# 	cif = row['cif']
# 	# fig, ax = plt.subplots(1,2)
# 	s = Structure.from_str(cif, fmt='cif')
# 	comp_dict = Composition(s.formula).get_el_amt_dict()
# 	full_formula = s.formula
# 	# ax[0].imshow(s.distance_matrix)
# 	# dm = s.distance_matrix
# 	sites = [s[l] for l in range(s.num_sites)]
# 	norms=np.linalg.norm(s.fractional_distance_matrix, ord=2, axis=1)
# 	j = 0
# 	sorted_ids = []
# 	for e,n in comp_dict.items():
# 		# print(e, n)
# 		sorted_ids.extend(list((np.argsort(norms[j:j+int(n)])+j)[:int(n/z)]))
# 		j=j+int(n)
# 	for k in range(int(s.num_sites/z)):
# 		s[k] = sites[sorted_ids[k]]
# 	s.remove_sites(range(int(s.num_sites/z), s.num_sites))
# 	dm = s.fractional_distance_matrix
# 	dm_resized[:dm.shape[0],:dm.shape[1]] = dm
# 	# dm_resized = sparse.csr_matrix(dm_resized)
# 	# ax[1].imshow(dm)
# 	# plt.show()
# 	# print(full_formula, Composition(s.formula).get_integer_formula_and_factor()[0], dm.shape)
# 	# dm_resized = cv2.resize(dm, dsize=(10, 10))
# 	dist_mat_list_tr.append(dm_resized)

# for i, row in df_test.iterrows():
# 	dm_resized = np.zeros((40, 40))
# 	# if row.Z<=32:
# 	z=row.Z
# 	cif = row['cif']
# 	# fig, ax = plt.subplots(1,2)
# 	s = Structure.from_str(cif, fmt='cif')
# 	comp_dict = Composition(s.formula).get_el_amt_dict()
# 	full_formula = s.formula
# 	# ax[0].imshow(s.distance_matrix)
# 	# dm = s.distance_matrix
# 	sites = [s[l] for l in range(s.num_sites)]
# 	norms=np.linalg.norm(s.fractional_distance_matrix, ord=2, axis=1)
# 	j = 0
# 	sorted_ids = []
# 	for e,n in comp_dict.items():
# 		# print(e, n)
# 		sorted_ids.extend(list((np.argsort(norms[j:j+int(n)])+j)[:int(n/z)]))
# 		j=j+int(n)
# 	for k in range(int(s.num_sites/z)):
# 		s[k] = sites[sorted_ids[k]]
# 	s.remove_sites(range(int(s.num_sites/z), s.num_sites))
# 	dm = s.fractional_distance_matrix
# 	dm_resized[:dm.shape[0],:dm.shape[1]] = dm
# 	# dm_resized = sparse.csr_matrix(dm_resized)
# 	# ax[1].imshow(dm)
# 	# plt.show()
# 	# print(full_formula, Composition(s.formula).get_integer_formula_and_factor()[0], dm.shape)
# 	# dm_resized = cv2.resize(dm, dsize=(10, 10))
# 	dist_mat_list_te.append(dm_resized)

# dist_mat_tr = np.array(dist_mat_list_tr)
# dist_mat_te = np.array(dist_mat_list_te)

# print(dist_mat_tr.shape, dist_mat_te.shape)

# np.save('data/databases/MP_formation_energy/dist_mat_tr_fractional_dis_mat_40x40_padded.npy', dist_mat_tr)
# np.save('data/databases/MP_formation_energy/dist_mat_te_fractional_dis_mat_40x40_padded.npy', dist_mat_te)

# with open('data/databases/MP_formation_energy/dis_mat_tr_500x500.pkl', 'wb') as f:
# 	pickle.dump(dist_mat_list_tr, f)

# with open('data/databases/MP_formation_energy/dis_mat_te_500x500.pkl', 'wb') as f:
# 	pickle.dump(dist_mat_list_te, f)




# def get_attributes(row):
# 	cif = row['cif']
# 	s = Structure.from_str(cif, fmt='cif')
# 	nsites = s.num_sites
# 	int_comp = Composition(Composition(s.formula).get_integer_formula_and_factor()[0])
# 	num_atoms_int_form = int_comp.num_atoms
# 	Z = nsites/num_atoms_int_form
	
# 	dm = np.pad(dm, [(0, 450-dm.shape[0]), (0, 450-dm.shape[0])], mode='constant')
# 	dist_mat_list.append(dm)
# 	return int_comp.formula, Z, nsites
# 	return dm_resized


# df_train['integer_formula'], df_train['Z'], df_train['nsites'] = zip(*df_train.apply(get_attributes, axis=1))
# df_test['integer_formula'], df_test['Z'], df_test['nsites'] = zip(*df_test.apply(get_attributes, axis=1))

# df_train['distance_matrix'] = df_train.apply(get_attributes, axis=1)
# df_test['distance_matrix'] = df_test.apply(get_attributes, axis=1)



# print(df_train['nsites'].max())
# print(df_train['Z'].max())

# print(df_test['nsites'].max())
# print(df_test['Z'].max())


# df_train = df_train[df_train.Z<=32]

# print(df_train)



# dist_mat_list = np.array(dist_mat_list)
# print(dist_mat_list.shape)

# df_train.to_csv('train_with_Z.csv', index=False)
# df_test.to_csv('test_with_Z.csv', index=False)





class ZNet(object):
	def __init__(self, 
				total_elements=103,
				n_classes=32,
				):
		self.total_elements = total_elements
		self.n_classes = n_classes
		self.efCls = ElementFraction()

	def build_ZNet(self, input_shape):
		# input_shape = (self.X_train.shape[1], 1)

		inputs = tf.keras.Input(shape=input_shape)

		x = layers.Conv1D(64,3, activation='relu')(inputs)
		x = layers.Conv1D(128,3, activation='relu')(x)
		x = layers.Conv1D(256,3, activation='relu')(x)
		# x = layers.Conv1D(256,3, activation='relu')(x)
		# x = layers.Conv1D(256,3, activation='relu')(x)
		x = layers.GlobalMaxPooling1D()(x)

		x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x)
		x = layers.Dropout(0.1)(x)

		x = layers.Dense(1024, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x)
		x = layers.Dropout(0.1)(x)

		# x = layers.Concatenate()([x, layers.Flatten()(x3)])

		x = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x)
		x = layers.Dropout(0.1)(x)

		output = layers.Dense(self.n_classes, activation='softmax')(x)
		model = tf.keras.Model([inputs], [output], name="ZNet")
		return model


	def parse_formulae(self, data_path=None, data=None):
		if data_path is not None:
			df = pd.read_csv(data_path, keep_default_na = False)
		if data is not None:
			df = data
		df = df[df.Z <= self.n_classes]
		print(df.info())
		df_a = df.apply(lambda x: self.efCls.featurize(Composition(x.formula)), axis=1, result_type='expand')
		df_featurized = pd.concat([df, df_a],axis='columns')
		X = df_featurized.drop(['ID', 'formula', 'integer_formula', 'Z', 'nsites', 'target', 'cif'], axis=1, errors='ignore')
		n_samples = df.shape[0]
		Y = np.zeros((n_samples, self.n_classes))
		y_cls = np.array(df_featurized['Z']).astype(int)
		Y[range(n_samples), y_cls-1] = 1

		# X_frac = self.generate_Z_frac(X)

		return np.expand_dims(X, axis=-1).astype('float32'), np.array(Y, dtype=np.float)

	def train_ZNet(self, train_path=None, val_path=None, data=None, lr=0.0001, batch_size=64, epochs=800, random_state=10):
		print("Loading and processing data ...")
		if (train_path and val_path):
			X_train, y_train = self.parse_formulae(data_path=train_path)
			X_val, y_val = self.parse_formulae(data_path=val_path)

		elif data is not None:
			X_data, y_data = self.parse_formulae(data=data)
			X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.15, random_state=random_state)
			# X_train_frac, X_val_frac, _, _ = train_test_split(X_data_frac, y_data, test_size=0.15, random_state=random_state)

		else:
			raise ValueError("Data path or a dataframe containing the data should be provided")

		input_shape = (X_train.shape[1], X_train.shape[2])

		# Z_frac_shape = (X_train_frac.shape[1], X_train_frac.shape[2])

		model_cnn = self.build_ZNet(input_shape)
		# model_dense = self.build_model(with_cnn=False)
		print(model_cnn.summary())
		model_cnn.compile(
					loss='categorical_crossentropy',
					optimizer=tf.keras.optimizers.Adam(learning_rate=lr), #tfa.optimizers.LAMB(learning_rate=lr),
					metrics=['accuracy'])

		

		checkpointer_cnn = tf.keras.callbacks.ModelCheckpoint('saved_models/best_model_cnn_Z.h5', monitor='accuracy', save_best_only=True)#attn_conv was best model 0.0294
		# checkpointer_dense = tf.keras.callbacks.ModelCheckpoint('saved_models/Rex_best_model_dense_clipped.h5', save_best_only=True)
		try:
			# history_cnn = model_cnn.fit([np.expand_dims(np.array(self.X_train), axis=-1), np.expand_dims(np.array(self.X_train_frac), axis=-1)], self.y_train, validation_split=0.15, epochs=epochs, batch_size=batch_size, callbacks=[checkpointer_cnn])
			# history_cnn = model_cnn.fit([np.expand_dims(np.array(self.X_train_wavelet), axis=-1)], self.y_train, validation_split= 0.15, epochs=epochs, batch_size=batch_size, callbacks=[checkpointer_cnn])			
			# history_cnn = model_cnn.fit(np.expand_dims(np.array(self.X_train)[:10000], axis=-1), self.y_train[:10000], validation_data=(np.expand_dims(np.array(self.X_test), axis=-1), self.y_test), epochs=epochs, batch_size=batch_size, callbacks=[checkpointer_cnn])
			history_cnn = model_cnn.fit([X_train], y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, callbacks=[checkpointer_cnn])
		except:
			model = tf.keras.models.load_model('saved_models/best_model_cnn_Z.h5')
			# evaluation = model.evaluate(X_val, y_val)
			predictions = np.argmax(model.predict(X_val), axis=1)
			targets = np.argmax(y_val, axis=1)
			df = pd.DataFrame({'target': targets, 'prediction': predictions})
			# print(df.iloc[100:])
			# print(evaluation)

	def test_ZNet(self, model_path, test_path, n_test=100):
		model = tf.keras.models.load_model(model_path, compile=False)

		X_test, y_test = self.parse_formulae(data_path=test_path)

		y_pred = model.predict([X_test])
		# print(df_test[['integer_formula', 'nsites']].iloc[20])
		# print(y_pred[20])
		# print(y_test[20])


class DisMatNet(ZNet):
	def __init__(self,
				dis_mat_shape=(10, 10),
				n_classes=32):
		self.dis_mat_shape = dis_mat_shape
		self.n_classes = n_classes
		self.efCls = ElementFraction()
		# super().__init__(**kwargs)
		self.f =  MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie"),
                         cf.ValenceOrbital(props=['avg']), cf.IonProperty(fast=True)])

	def build_DisMatNet(self, input_shape):
		inputs = tf.keras.Input(shape=input_shape)

		x1 = layers.Conv1D(64,3, activation='relu')(inputs)
		x2 = layers.Conv1D(128,3, activation='relu')(x1)
		x = layers.Conv1D(256,3, activation='relu')(x2)
		x = layers.Conv1D(256,3, activation='relu')(x)
		x = layers.GlobalMaxPooling1D()(x)
		# x = layers.Flatten()(x)

		# x = layers.Dense(256, activation='relu')(x)
		# x = layers.Concatenate()([x, layers.Flatten()(x2)])
		# x = layers.Dense(1024, activation='relu')(x)
		# x = layers.Concatenate()([x, layers.Flatten()(x1)])
		# x = layers.Dense(1024, activation='relu')(x)
		# x = layers.Dense(2048, activation='relu')(x)
		# x = layers.Concatenate()([x, layers.Flatten()(x1)])
		# x = layers.Dense(512, activation='relu')(x)
		# output = layers.Dense(780, activation='linear')(x)
		# outs = []
		# for i in range(780):
		# 	# y = layers.Conv1D(10,3, activation='relu')(inputs)
		# 	# y = layers.GlobalMaxPooling1D()(y)
		# 	# y = layers.Dense(256, activation='relu')(inputs)
		# 	# y = layers.Dense(128, activation='relu')(inputs)
		# 	# y = layers.Dense(256, activation='relu')(y)
		# 	y = layers.Dense(64, activation='relu')(inputs)
		# 	y_out = layers.Dense(1)(y)
		# 	outs.append(y_out)
		# print(outs)
		# print(len(outs))
		# output = layers.Concatenate()(outs)

		

		x = layers.Dense(self.dis_mat_shape[0]*self.dis_mat_shape[1], activation='relu')(x)

		x = layers.Reshape((self.dis_mat_shape[0], self.dis_mat_shape[1], 1))(x)

		# x = tf.expand_dims(x, axis=-1)
		x3 = layers.Conv2D(64,3, activation='relu', padding='same')(x)
		x = layers.Conv2D(128,1, activation='relu', padding='same')(x3)
		# x = layers.Conv2D(256,1, activation='relu', padding='same')(x)
		x = layers.Conv2D(256,1, activation='relu', padding='same')(x)
		x = layers.Concatenate()([x, x3])
		# x = layers.Conv2D(256,3, activation='relu', padding='same')(x)
		output = layers.Conv2D(1, 1, activation='linear', padding='same')(x)

		model = tf.keras.Model([inputs], [output], name="DisMatNet")
		return model

	def parse_formulae(self, data_path=None, data=None):
		if data_path is not None:
			df = pd.read_csv(data_path, keep_default_na = False)
		if data is not None:
			df = data
		df['num_atoms'] = df['formula'].apply(lambda x: Composition(Composition(x).get_integer_formula_and_factor()[0]).num_atoms)
		# df = df[df.num_atoms <= 40]
		df_a = df.apply(lambda x: self.efCls.featurize(Composition(x.formula)), axis=1, result_type='expand')
		# df_a = df.apply(lambda x: self.f.featurize(Composition(x.formula)), axis=1, result_type='expand').fillna(0)
		df_featurized = pd.concat([df, df_a], axis='columns')
		X = df_featurized.drop(['ID', 'formula', 'integer_formula', 'nsites', 'target', 'Z', 'cif'], axis=1, errors='ignore')

		return np.expand_dims(X, axis=-1)#.astype('float32')
		# return np.array(X)
		
		# n_samples = df.shape[0]
		# Y = np.zeros((n_samples, self.n_classes))
		# y_cls = np.array(df_featurized['Z']).astype(int)
		# Y[range(n_samples), y_cls-1] = 1

	def train_DisMatNet(self, train_path=None, val_path=None, train_y_path=None, val_y_path=None, data=None, lr=0.0001, batch_size=64, epochs=800, random_state=10):
		print("Loading and processing data ...")
		if (train_path and val_path):
			X_train = self.parse_formulae(data_path=train_path)#[:2000]
			y_train = np.expand_dims(np.load(train_y_path), axis=-1)#/10.0
			
			def get_uptriangle(array):
				ids = np.triu_indices(array.shape[0], k=1)
				return array[ids]
			# y_train = np.load(train_y_path)#/10.0
			# y_train = np.array(list(map(get_uptriangle,y_train)))#[:2000]
			X_val = self.parse_formulae(data_path=val_path)#[:2000]
			y_val = np.expand_dims(np.load(val_y_path), axis=-1)#/10.0
			# y_val = np.load(val_y_path)#/10.0
			# y_val = np.array(list(map(get_uptriangle,y_val)))#[:2000]

		# elif data is not None:
		# 	X_data, y_data = self.parse_formulae(data=data)
		# 	X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.15, random_state=random_state)
		# 	# X_train_frac, X_val_frac, _, _ = train_test_split(X_data_frac, y_data, test_size=0.15, random_state=random_state)

		# else:
		# 	raise ValueError("Data path or a dataframe containing the data should be provided")

		input_shape = (X_train.shape[1], X_train.shape[2])
		# input_shape = (X_train.shape[1], )
		# # Z_frac_shape = (X_train_frac.shape[1], X_train_frac.shape[2])

		model_cnn = self.build_DisMatNet(input_shape)
		# model_dense = self.build_model(with_cnn=False)
		print(model_cnn.summary())
		model_cnn.compile(
					loss='mae',
					optimizer=tf.keras.optimizers.Adam(learning_rate=lr), #tfa.optimizers.LAMB(learning_rate=lr),
					metrics=['mse', 'mae'])

		# ### uncomment for ML
		# reg = MultiOutputRegressor(RandomForestRegressor(random_state=9, n_estimators=8, max_depth=32, verbose=3))
		# # reg = MultiOutputRegressor(RandomForestRegressor(random_state=9, verbose=3))
		# reg.fit(X_train.reshape(X_train.shape[0], X_train.shape[1]), y_train)
		# # reg.fit(X_train.reshape(X_train.shape[0], X_train.shape[1]), y_train.reshape(y_train.shape[0], y_train.shape[1]*y_train.shape[2]))
		# predicted_distances = reg.predict(X_val.reshape(X_val.shape[0], X_val.shape[1]))
		# # mae = mean_absolute_error(y_val.reshape(y_val.shape[0], y_val.shape[1]*y_val.shape[2]), predicted_distances)
		# mae = mean_absolute_error(y_val, predicted_distances)
		# print(mae)
		# y_test = y_val
		# print(predicted_distances.shape, y_test.shape)
		# filename = 'data/databases/MP_formation_energy/random_forest_full.joblib.pkl'
		# joblib.dump(reg, filename, compress=3)
		# for i in range(30):
		# 	pred = predicted_distances[i+20]
		# 	y = y_test[i+20]
		# 	# print(np.c_[pred[:50], y[:50]],'\n\n')
			
		# 	# print(np.c_[pred, y_test],'\n\n')
		# 	l = len(y)
		# 	pred = pred[:l]
		# 	print(pred.shape, y.shape)
		# 	size_X = int(np.sqrt(2*l))+1
		# 	# put it back into a 2D symmetric array
		# 	# size_X = 3
		# 	X = np.zeros((size_X,size_X))
		# 	X[np.triu_indices(X.shape[0], k = 1)] = pred
		# 	pred = X + X.T

		# 	X = np.zeros((size_X,size_X))
		# 	X[np.triu_indices(X.shape[0], k = 1)] = y
		# 	y = X + X.T
			
		# 	fig, ax = plt.subplots(1,2)
		# 	# im=ax[0].imshow(y_test[60+i].reshape(self.dis_mat_shape))
		# 	# im=ax[1].imshow(y_pred[60+i].reshape(self.dis_mat_shape))
		# 	im = ax[0].imshow(y)
		# 	im = ax[1].imshow(pred)
		# 	fig.subplots_adjust(right=0.8)
		# 	cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
		# 	fig.colorbar(im, cax=cbar_ax)
		# 	plt.show()



		# for i in range(10):
		# 	fig, ax = plt.subplots(1,2)
		# 	# im=ax[0].imshow(y_test[60+i].reshape(self.dis_mat_shape))
		# 	# im=ax[1].imshow(y_pred[60+i].reshape(self.dis_mat_shape))
		# 	N = int(X_val[90+i][0])
		# 	im=ax[0].imshow(np.squeeze(y_val[90+i][:N, :N]))
		# 	im=ax[1].imshow(np.squeeze(pred[90+i].reshape(40, 40)[:N, :N]))
		# 	fig.subplots_adjust(right=0.8)
		# 	cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
		# 	fig.colorbar(im, cax=cbar_ax)

		# 	plt.show()

		checkpointer_cnn = tf.keras.callbacks.ModelCheckpoint('saved_models/best_model_cnn_DisMat.h5', monitor='mae', save_best_only=True)#attn_conv was best model 0.0294
		# checkpointer_dense = tf.keras.callbacks.ModelCheckpoint('saved_models/Rex_best_model_dense_clipped.h5', save_best_only=True)
		try:
			# history_cnn = model_cnn.fit([np.expand_dims(np.array(self.X_train), axis=-1), np.expand_dims(np.array(self.X_train_frac), axis=-1)], self.y_train, validation_split=0.15, epochs=epochs, batch_size=batch_size, callbacks=[checkpointer_cnn])
			# history_cnn = model_cnn.fit([np.expand_dims(np.array(self.X_train_wavelet), axis=-1)], self.y_train, validation_split= 0.15, epochs=epochs, batch_size=batch_size, callbacks=[checkpointer_cnn])			
			# history_cnn = model_cnn.fit(np.expand_dims(np.array(self.X_train)[:10000], axis=-1), self.y_train[:10000], validation_data=(np.expand_dims(np.array(self.X_test), axis=-1), self.y_test), epochs=epochs, batch_size=batch_size, callbacks=[checkpointer_cnn])
			history_cnn = model_cnn.fit([X_train], y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, callbacks=[checkpointer_cnn])
		except:
			model = tf.keras.models.load_model('saved_models/best_model_cnn_DisMat.h5')
			# evaluation = model.evaluate(X_val, y_val)
			predictions = np.argmax(model.predict(X_val), axis=1)
			targets = np.argmax(y_val, axis=1)
			df = pd.DataFrame({'target': targets, 'prediction': predictions})

	def test_DisMatNet(self, model_path, test_path, test_y_path, n_test=30):
		model = tf.keras.models.load_model(model_path)
		# model  = joblib.load('data/databases/MP_formation_energy/random_forest.joblib.pkl') 

		X_test = self.parse_formulae(data_path=test_path)
		y_test = np.load(test_y_path)#/10.0
		y_pred = model.predict([X_test])
		# y_pred = model.predict(X_test)

		# def get_uptriangle(array):
		# 	ids = np.triu_indices(array.shape[0], k=1)
		# 	return array[ids]
		# y_test = np.array(list(map(get_uptriangle,y_test)))


		# for i in range(40):
		# 	pred = y_pred[i+20]
		# 	y = y_test[i+20]
			
		# 	# print(np.c_[pred, y_test],'\n\n')
		# 	l = len(y)
		# 	pred = pred[:l]
		# 	print(np.c_[pred[:50], y[:50]],'\n\n')
		# 	size_X = int(np.sqrt(2*l))+1
		# 	# put it back into a 2D symmetric array
		# 	# size_X = 3
		# 	X = np.zeros((size_X,size_X))
		# 	X[np.triu_indices(X.shape[0], k = 1)] = pred
		# 	pred = X + X.T

		# 	X = np.zeros((size_X,size_X))
		# 	X[np.triu_indices(X.shape[0], k = 1)] = y
		# 	y = X + X.T
			
		# 	fig, ax = plt.subplots(1,2)
		# 	# im=ax[0].imshow(y_test[60+i].reshape(self.dis_mat_shape))
		# 	# im=ax[1].imshow(y_pred[60+i].reshape(self.dis_mat_shape))
		# 	im = ax[0].imshow(y)
		# 	im = ax[1].imshow(pred)
		# 	fig.subplots_adjust(right=0.8)
		# 	cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
		# 	fig.colorbar(im, cax=cbar_ax)
		# 	plt.show()

		for i in range(n_test):
			fig, ax = plt.subplots(1,2)
			# im=ax[0].imshow(y_test[60+i].reshape(self.dis_mat_shape))
			# im=ax[1].imshow(y_pred[60+i].reshape(self.dis_mat_shape))
			N = int(X_test[90+i][0])
			im=ax[0].imshow(np.squeeze(y_test[90+i][:N, :N]))
			im=ax[1].imshow(np.squeeze(y_pred[90+i][:N, :N]))
			fig.subplots_adjust(right=0.8)
			cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
			fig.colorbar(im, cax=cbar_ax)

			plt.show()






			

if __name__=='__main__':
	znet = ZNet()
	dismatnet = DisMatNet()
	# znet.train_ZNet(train_path='data/databases/MP_formation_energy/train_with_Z.csv', val_path='data/databases/MP_formation_energy/test_with_Z.csv', epochs=100)
	# znet.test_ZNet(model_path='saved_models/best_model_cnn_Z.h5', test_path='data/databases/MP_formation_energy/test_with_Z.csv')

	# dismatnet.train_DisMatNet(train_path='data/databases/MP_formation_energy/train_with_Z.csv',
	# 						 val_path='data/databases/MP_formation_energy/test_with_Z.csv', 
	# 						 train_y_path='data/databases/MP_formation_energy/dist_mat_tr_fractional_dis_mat_10x10.npy',
	# 						 val_y_path='data/databases/MP_formation_energy/dist_mat_te_fractional_dis_mat_10x10.npy',
	# 						 epochs=300)
	dismatnet.test_DisMatNet(model_path='saved_models/best_model_cnn_DisMat.h5', 
							test_path='data/databases/MP_formation_energy/test_with_Z.csv', 
							test_y_path='data/databases/MP_formation_energy/dist_mat_te_fractional_dis_mat_10x10.npy')

'''