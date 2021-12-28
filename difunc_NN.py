
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
from matplotlib import rcParams

from pymatgen import Composition, Element, MPRester
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from matminer.featurizers.composition import ElementFraction

import itertools

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tensorflow.keras import backend as K
config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

# print(rcParams.keys())
rcParams.update({'figure.autolayout': True})
rcParams["font.family"] = "Times New Roman"
rcParams['axes.linewidth'] = 2
rcParams['xtick.major.size'] = 6
rcParams['xtick.major.width'] = 1.5
rcParams['xtick.minor.size'] = 4
rcParams['xtick.labelsize'] = 16
rcParams['ytick.major.size'] = 6
rcParams['ytick.major.width'] = 1.5
rcParams['ytick.minor.size'] = 4
rcParams['ytick.labelsize'] = 16

# reproducibility
np.random.seed(123)

class DiFuncNN(object):
	def __init__(self):
		## self.df = pd.read_pickle('difunc_data/labelled_difunc_data_opt.pkl')
		## self.df = pd.read_pickle('difunc_data/df_X.pkl')
		## self.df['red_formula'] = self.df.apply(lambda x: Composition(x.full_formula).reduced_formula, axis=1)
		## self.df_n = self.df.drop_duplicates(subset=['red_formula'], keep='first')
		## ef = ElementFraction()
		## a_df = self.df_n.apply(lambda x: ef.featurize(Composition(x.full_formula)), axis=1, result_type='expand')
		## df = pd.concat([self.df_n[['full_formula']], a_df], axis='columns')
		## df.to_pickle('difunc_data/X_elem_frac.pkl')
		
		## print(self.df_n.info())

		# # print(self.df.info())

		# # self.X = np.expand_dims(np.load('difunc_data/X.npy'), axis=-1)
		# # self.X = np.load('difunc_data/X.npy')

		# self.columns = np.load('difunc_data/columns.npy')
		# self.df_X = pd.read_pickle('difunc_data/X.pkl')
		# self.columns = range(0,103)
		self.df_X = pd.read_pickle('difunc_data/X_elem_frac.pkl')
		self.columns = range(self.df_X.shape[1]-1)
		# # self.df_X = self.df_X.drop_duplicates(subset=self.columns, keep="first")
		# # self.df_X.info()
		## x = self.df_n[['full_formula']+ list(self.columns)]
		## x.to_pickle('difunc_data/X.pkl')




		idx = list(self.df_X.index)
		self.Rex = np.load('difunc_data/Rex_c.npy')
		self.Rex = self.Rex[idx, :]
		self.Imx = np.load('difunc_data/Imx_c.npy')
		self.Imx = self.Imx[idx, :]
		
		# print(self.df_X.shape)
		# print(self.Rex.shape)
		# print(self.Imx.shape)
		# print(np.unique(self.X, axis=0).shape)
		# print(self.df_X[self.columns])
		# self.Eng = np.load('difunc_data/Eng.npy')
		# self.max_energy = np.max(self.Eng, axis=1)

		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df_X, self.Imx, test_size=0.1, random_state=10)

	def featurize(self, df):
		def get_compostion(c): # Function to get compositions from chemical formula using pymatgen
		    try:
		        return Composition(c)
		    except:
		        return None

		f =  MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie"),
                         cf.ValenceOrbital(props=['avg']), cf.IonProperty(fast=True)])
		# df = self.df
		df = df.dropna(subset=['full_formula'])
		df['composition'] = df['full_formula'].apply(get_compostion)

		x = np.array(f.featurize_many(df['composition']))
		df_x = pd.DataFrame(x, columns=range(x.shape[1]))
		# df_x = df_x.dropna(axis=1)

		df.reset_index(drop=True, inplace=True)
		df_x.reset_index(drop=True, inplace=True)
		df_concat = pd.concat( [df, df_x], axis=1) 
		# print(df_concat.info())

		# df_concat.to_pickle('difunc_data/df_X.pkl')

		return df_concat

	def vectorize(self, df):
		ef = ElementFraction()
		a_df = df.apply(lambda x: ef.featurize(Composition(x.full_formula)), axis=1, result_type='expand')
		if (set(['full_formula', 'formation_energy_per_atom', 'e_above_hull', 'red_formula']).issubset(df.columns.tolist())):
			df_vec = pd.concat([df[['full_formula', 'formation_energy_per_atom', 'e_above_hull', 'red_formula']], a_df], axis='columns')
		else:
			df_vec = pd.concat([df[['full_formula']], a_df], axis='columns')

		return df_vec


		# print(df_x.info())
		# columns = np.array(df_x.columns)
		# np.save('difunc_data/columns.npy', columns)
		# print(np.array(df_x.columns))
		# x = x[:, ~np.isnan(x).any(axis=0)]
		# realx = np.stack(np.array(df['opt_realxx']), axis=0)
		# imagx = np.stack(np.array(df['opt_imagxx']), axis=0)
		# energy = np.stack(np.array(df['opt_en']), axis=0)

		# print(x.shape, realx.shape, imagx.shape, energy.shape)

		# np.save('difunc_data/X.npy', x)
		# np.save('difunc_data/Rex.npy', realx)
		# np.save('difunc_data/Imx.npy', imagx)
		# np.save('difunc_data/Eng.npy', energy)

	def fix_energy_range(self, max_energy=30, n_samples=3000):
		Rex = []
		Imx = []
		for i in range(self.Eng.shape[0]):	# iterate through samples
			eng_clipped = self.Eng[i][self.Eng[i]<=max_energy] # energy less than max_energy specified
			rex_clipped = self.Rex[i][:len(eng_clipped)]
			imx_clipped = self.Imx[i][:len(eng_clipped)]
			# print(eng_clipped.shape, rex_clipped.shape)
			e_samples = np.linspace(0, max_energy, n_samples)
			rex_interp = np.interp(e_samples, eng_clipped, rex_clipped)
			imx_interp = np.interp(e_samples, eng_clipped, imx_clipped)

			Rex.append(rex_interp)
			Imx.append(imx_interp)

			# plt.plot(eng_clipped, imx_clipped, c='b', label='Truth')
			# plt.plot(e_samples, imx_interp, c='r', label='Interpolated')
			# plt.legend()
			# plt.show()
		print(np.array(Rex).shape, np.array(Imx).shape)
		np.save('difunc_data/Rex_c.npy', Rex)
		np.save('difunc_data/Imx_c.npy', Imx)


	def difunc_net(self, with_cnn=True):
		input_shape = (len(self.columns),1)

		inputs = tf.keras.Input(shape=input_shape)

		if with_cnn:
			x = layers.Conv1D(64,3, activation='relu')(inputs)
			x = layers.Conv1D(128,3, activation='relu')(x)
			x = layers.Conv1D(256,3, activation='relu')(x)
			x = layers.Conv1D(256,3, activation='relu')(x)
			x_branch = layers.Flatten()(x)
		else:
			x = layers.Flatten()(inputs)
			x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x)
			x = layers.Dropout(0.1)(x)

			x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x)
			x = layers.Dropout(0.1)(x)

			x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x)
			x = layers.Dropout(0.1)(x)

			x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x)
			x_branch = layers.Dropout(0.1)(x)


		x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x_branch)
		x = layers.Dropout(0.1)(x)

		x = layers.Dense(1024, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x)
		x = layers.Dropout(0.1)(x)

		x = layers.Dense(1024, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x)
		x = layers.Dropout(0.1)(x)

		x = layers.Dense(2048, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x)
		x = layers.Dropout(0.1)(x)

		# x = layers.Dense(2048, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x)
		# x = layers.Dropout(0.1)(x)

		# x = layers.Dense(2048, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x)
		# x = layers.Dropout(0.1)(x)

		x = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-8), activation='relu')(x)
		x = layers.Dropout(0.1)(x)
		spectrum_out = layers.Dense(self.Rex.shape[1])(x)


		# y = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x_branch)
		# y = layers.Dropout(0.1)(y)

		# y = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(y)
		# y = layers.Dropout(0.1)(y)

		# y = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(y)
		# y = layers.Dropout(0.1)(y)

		# max_energy_out = layers.Dense(1, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(y)



		#

		model = tf.keras.Model([inputs], [spectrum_out], name="difunc_net")
		return model



	def build_cnn(self):
		input_shape = (self.X_train.shape[1]-1,1)

		inputs = tf.keras.Input(shape=input_shape)

		x1 = layers.Conv1D(64,3, activation='relu')(inputs)
		x2 = layers.Conv1D(128,3, activation='relu')(x1)
		x3 = layers.Conv1D(256,3, activation='relu')(x2)
		x = layers.Conv1D(256,3, activation='relu')(x3)
		x = layers.Conv1D(256,3, activation='relu')(x)

		x = layers.Flatten()(x)

		x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x)
		x = layers.Dropout(0.1)(x)

		x = layers.Dense(1024, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x)
		x = layers.Dropout(0.1)(x)

		# x = layers.Concatenate()([x, layers.Flatten()(x3)])

		x = layers.Dense(1024, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x)
		x = layers.Dropout(0.1)(x)

		x = layers.Concatenate()([x, layers.Flatten()(x2)])

		x = layers.Dense(2048, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x)
		x = layers.Dropout(0.1)(x)

		# x = layers.Dense(2048, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x)
		# x = layers.Dropout(0.1)(x)
		x = layers.Concatenate()([x, layers.Flatten()(x1)])

		x = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-8), activation='relu')(x)
		# x = layers.Dropout(0.1)(x)

		out = layers.Dense(3000, activation='linear')(x)
		model = tf.keras.Model([inputs], [out], name="difunc_net")
		return model

	def difunc_net_trail(self):
		input_shape = (len(self.columns),1)

		inputs = tf.keras.Input(shape=input_shape)

		x1 = layers.Conv1D(64,3, activation='relu')(inputs)
		x2 = layers.Conv1D(128,3, activation='relu')(x1)
		x3 = layers.Conv1D(256,3, activation='relu')(x2)
		x = layers.Conv1D(256,3, activation='relu')(x3)
		x = layers.Flatten()(x)

		x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x)
		x = layers.Dropout(0.1)(x)

		x = layers.Dense(1024, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x)
		x = layers.Dropout(0.1)(x)

		# x = layers.Concatenate()([x, layers.Flatten()(x3)])

		x = layers.Dense(1024, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x)
		x = layers.Dropout(0.1)(x)

		x = layers.Concatenate()([x, layers.Flatten()(x2)])

		x = layers.Dense(2048, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x)
		x = layers.Dropout(0.1)(x)

		# x = layers.Dense(2048, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x)
		# x = layers.Dropout(0.1)(x)

		# x = layers.Dense(2048, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x)
		# x = layers.Dropout(0.1)(x)
		x = layers.Concatenate()([x, layers.Flatten()(x1)])

		x = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-8), activation='relu')(x)
		x = layers.Dropout(0.1)(x)

		spectrum_out = layers.Dense(self.Rex.shape[1])(x)
		model = tf.keras.Model([inputs], [spectrum_out], name="difunc_net")
		return model



	def train(self, lr=0.0001, batch_size=24, epochs=300):
		model_cnn = self.difunc_net(with_cnn=True)
		# model_cnn = self.difunc_net_trail()
		model_cnn = self.build_cnn()
		# model_dense = self.difunc_net(with_cnn=False)
		model_cnn.compile(
					loss="mae",
					optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=['mae'])
		# model_dense.compile(
		# 			loss="mae",
		# 			optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=['mae'])

		checkpointer_cnn = tf.keras.callbacks.ModelCheckpoint('saved_models/Imx_best_model_skipcnn_clipped.h5', save_best_only=True)
		# checkpointer_dense = tf.keras.callbacks.ModelCheckpoint('saved_models/Rex_best_model_dense_clipped.h5', save_best_only=True)
		try:
			history_cnn = model_cnn.fit(np.expand_dims(np.array(self.X_train[list(self.columns)]), axis=-1), self.y_train, validation_split=0.15, epochs=epochs, batch_size=batch_size, callbacks=[checkpointer_cnn])
			# history_dense = model_dense.fit(np.expand_dims(np.array(self.X_train[list(self.columns)]), axis=-1), self.y_train, validation_split=0.15, epochs=epochs, batch_size=batch_size, callbacks=[checkpointer_dense])
			# self.plot_training_history(history_cnn, history_dense, epochs)

		except KeyboardInterrupt:
			print('############# Training interrupted ###########')

	def plot_training_history(self, history_cnn, history_dense, epochs):
		fig, ax = plt.subplots(1,2,figsize=(14,6))
		ax[0].plot(range(1, epochs+1), history_cnn.history['loss'], c='b', lw='2', ls='-')
		ax[0].plot(range(1, epochs+1),history_cnn.history['val_loss'], c='r', lw='2', ls='-')
		ax[0].set_ylim([0.2, 2.4])
		ax[0].set_title('With CNN model loss', fontsize=20)
		ax[0].set_ylabel('MAE', fontsize=18)
		ax[0].set_xlabel('epoch', fontsize=18)
		ax[0].legend(['train', 'validation'], loc='upper left', fontsize=18)

		ax[1].plot(range(1, epochs+1),history_dense.history['loss'], c='b', lw='2', ls='-')
		ax[1].plot(range(1, epochs+1),history_dense.history['val_loss'], c='r', lw='2', ls='-')
		ax[1].set_ylim([0.2, 2.4])
		ax[1].set_title('Without CNN model loss', fontsize=20)
		ax[1].set_ylabel('MAE', fontsize=18)
		ax[1].set_xlabel('epoch', fontsize=18)
		ax[1].legend(['train', 'validation'], loc='upper left', fontsize=18)

		fig.savefig('training_history_clipped_Rex.png', dpi=100)


		plt.show()


	def test(self, n_test=100):
		model = tf.keras.models.load_model('saved_models/Imx_best_model_skipcnn_clipped.h5', compile=False)
		y_pred = model.predict(np.expand_dims(np.array(self.X_test[list(self.columns)]), axis=-1))
		for i in range(n_test):
			fig, ax = plt.subplots(1,1,figsize=(8,6))
			ax.plot(np.linspace(0, 30, 3000), y_pred[10+i], label='Prediction', c='r', lw=2)
			ax.plot(np.linspace(0, 30, 3000), self.y_test[10+i], label='Ground Truth', c='b', lw=2, ls='--')
			ax.legend(fontsize=18)
			ax.set_xlabel('Energy (eV)', fontsize=18)
			ax.set_ylabel("Imaginary Permittivity", fontsize=18)
			composition = self.X_test.iloc[10+i]['full_formula']
			ax.set_title(composition+ ' dielectric function')
			fig.savefig('test_results_Imx/test_'+str(i)+'.png', dpi=100)
			# plt.show()
			plt.close()
		mae = mean_absolute_error(self.y_test, y_pred)
		mape = mean_absolute_percentage_error(self.y_test, y_pred)
		mse = mean_squared_error(self.y_test, y_pred)
		r2 = r2_score(self.y_test, y_pred)
		# print(i)
		print('MAE: ', mae)
		print('MAPE: {:.3f}%'.format(mape))
		print('RMSE: ', np.sqrt(mse))
		print('R2 score: ', r2)
		print('\n\n')

	def predict_external(self, composition,  model_re, model_im):
		# def get_compostion(c): # Function to get compositions from chemical formula using pymatgen
		#     try:
		#         return Composition(c)
		#     except:
		#         return None

		# f =  MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie"),
  #                        cf.ValenceOrbital(props=['avg']), cf.IonProperty(fast=True)])
		# df = pd.DataFrame({'full_formula': [composition]})
		# df = df.dropna(subset=['full_formula'])
		# df['composition'] = df['full_formula'].apply(get_compostion)
		# # df = df.dropna(subset=['composition'])
		# columns = np.load('difunc_data/columns.npy')
		# x = np.array(f.featurize_many(df['composition']))
		# df_x = pd.DataFrame(x, columns=range(x.shape[1]))
		# df_x = df_x.dropna(axis=1)
		# X = np.array(df_x[list(columns)])

		''' elem frac'''
		df = pd.DataFrame({'full_formula': [composition]})
		dfv = self.vectorize(df)
		X = np.array(dfv[range(103)])
		
		y_pred_real = model_re.predict(X)
		y_pred_imag = model_im.predict(X)		
		plt.plot(np.linspace(0, 30, 3000), y_pred_real[0], lw=2, ls='--', c='purple', label='Real $\epsilon$')
		plt.plot(np.linspace(0, 30, 3000), y_pred_imag[0], lw=2, ls='--', c='red', label='Imaginary $\epsilon$')
		plt.legend()
		plt.xlim(0, 15)
		plt.title(composition + " - Dielectric constant vs frequency", fontsize=16)
		plt.savefig('ENZ_to_fabricate/'+composition + '.png', dpi=100)
		# plt.show()
		plt.close()
		# self.plot_prediction(composition, '~/QM/em_near_zero/nzi_discovery/DFT/tests/mp-1185263_LiPaO3', y_pred_real, y_pred_imag)

	def plot_prediction(self, composition, dft_path, y_pred_real, y_pred_imag):
		df_real = pd.read_csv(dft_path+"/real.dat", header=None, sep=' ').iloc[:2000].astype(np.float64)
		df_imag = pd.read_csv(dft_path+"/imag.dat", header=None, sep=' ').iloc[:2000].astype(np.float64)
		# print(df_real.info())

		plt.margins(x=0)
		plt.tight_layout()
		plt.plot(np.array(df_real.iloc[:, [0]]), np.array(df_real.iloc[:, [1]]), color='blue', linewidth=2, label='$\epsilon_r$')
		plt.plot(np.array(df_imag.iloc[:, [0]]), np.array(df_imag.iloc[:, [1]]), color='red', linewidth=2, label='$\epsilon_i$')

		plt.plot(np.linspace(0, 30, 3000), y_pred_real[0], lw=2, ls='--', c='blue', label='$\epsilon_r$ prediction')
		plt.plot(np.linspace(0, 30, 3000), y_pred_imag[0], lw=2, ls='--', c='red', label='$\epsilon_i$ prediction')

		# for wavelength plot
		# plt.plot(1239.8/np.linspace(0, 30, 3000)[::-1], y_pred_real[0][::-1], lw=2, ls='--', c='blue', label='$\epsilon_r$ prediction')
		# plt.plot(1239.8/np.linspace(0, 30, 3000)[::-1], y_pred_imag[0][::-1], lw=2, ls='--', c='red', label='$\epsilon_i$ prediction')

		plt.ylabel('Dielectric constant', fontsize=16)
		plt.xlabel('Frequency (eV)', fontsize=16)
		plt.legend(fontsize=14)
		plt.xlim(0, 15)
		# for wavelength
		# plt.xlim(min(1239.8/np.linspace(0, 30, 3000)[::-1]), 1500)

		# plt.ylim(-2,40)
		plt.title(composition + " - Dielectric constant vs frequency", fontsize=16)
		plt.savefig(composition + '_DFT_DL.png', dpi=100)
		plt.show()
		plt.close()

	def plot_Al_Ag(self):
		plasma_freq = [1.04, 0.97, 1.1, 1.3, 1.9, 2.2, 1.86, 1.62, 1.5, 1.7]
		imag_freq = [38.5, 35.12, 13, 11.6, 19.2, 15.7, 19.86, 25.6, 30.5, 34]
		plt.plot(np.linspace(0.05, 0.95, 10), imag_freq, lw=2, ls='--', c='purple')
		plt.xlim(0.05, 0.95)
		plt.xticks(np.linspace(0.05, 0.95, 10))
		plt.ylabel("Imaginary permittivity", fontsize=18)
		plt.xlabel('AlxAg(1-x)', fontsize=18)
		plt.savefig('Ag_Al/Al_Ag_imag.png', dpi=100)
		plt.show()




	# def difunc_net(self, input_shape=(131,)):

model_re = tf.keras.models.load_model('saved_models/Rex_best_model_skipcnn_clipped.h5', compile=False)
model_im = tf.keras.models.load_model('saved_models/Imx_best_model_skipcnn_clipped.h5', compile=False)

dfnn = DiFuncNN()
# dfnn.featurize()
# dfnn.train(epochs=500)
# dfnn.test(n_test=100)
# dfnn.fix_energy_range()

'''

comp_list = ['KCa4B3O9',
'NaZr2MnF11',
'SrCuSe2O6',
'Yb2Sn2O7',
'YbSeClO3',
'Y3InS6',
'LiCeF5',
'RbYbP2O7',
'Ca3Cu3P4O16',
'Ce3B2Cl3O6',
'UV2O6',
'Sr2CuOsO6',
'Na2MgFeF7',
'TaO2F',
'RbGaAgF6',
'Nd3U2O10',
'MgNiF6',
'Rb3CeF6',
'LiPaO3',
'Rb2EuF5',
'NaCa2Mg2V3O12',
'LiScVO4',
'Tl2Co2S3O12',
'Mg4CuO5',
'Rb3YbP2O8',
'Cu6SnO8',
'Ca4Ga3FeO10',

]
for c in comp_list:
	dfnn.predict_external(c, model_re, model_im)

# e=Element('H')
# metals = []
# semiconductors = []
# for i in range(1, 100):
# 	elem = e.from_Z(i)
# 	if elem.is_metal:
# 		metals.append(elem.name)
# 	elif elem.is_metalloid:
# 		semiconductors.append(elem.name)

# # print(len(metals))
# # print(len(semiconductors))
# elements = metals+semiconductors
# alloy_combinations = itertools.combinations(elements, 2)

# # print(len(list(alloy_combinations)))
# model_re = tf.keras.models.load_model('saved_models/Rex_best_model_cnn_clipped.h5', compile=False)
# model_im = tf.keras.models.load_model('saved_models/Imx_best_model_cnn_clipped.h5', compile=False)
# charge_bal_comps = []
# for alloy in alloy_combinations:
# 	for i in range(1, 10):
# 		comp = Composition('%s%s%s%s'%(alloy[0], str(i), alloy[1], str(10-i)))
# 		oxi_guess = comp.oxi_state_guesses()
# 		if len(oxi_guess)>0:
# 			print(comp.formula)
# 			charge_bal_comps.append(comp)
# 			# print('%s%s%s%s'%(alloy[0], str(i), alloy[1], str(10-i)))
# 			dfnn.predict_external(comp.formula.replace(" ", ""), model_re, model_im)
# 			# dfnn.predict_external('%s%s%s%s'%(alloy[0], str(i), alloy[1], str(10-i)))
# # dfnn.plot_Al_Ag()
# print(len(charge_bal_comps))

'''