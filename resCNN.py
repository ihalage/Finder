import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
# from robust_loss import adaptive
# from keras_self_attention import SeqSelfAttention

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler

from pymatgen import Composition, Element
from matminer.featurizers.composition import ElementFraction
from mendeleev import element
# from helper import ElemFeatures

from scipy import signal

import matplotlib.pyplot as plt
from matplotlib import rcParams

# from comp2periodicT import Comp2PeriodicT
# from sparse_layers import SparseConv2D, MaxPoolingWithArgmax2D
# from AttentionWithContext import AttentionWithContext
# from attention import Attention
# from attn_augconv import augmented_conv2d
# from seq_self_attention import SeqSelfAttention
# from CALCom import CALCom
from utils import Normalizer, NormalizeTensor

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
# np.random.seed(123)


## import tensorflow as tf
from tensorflow.keras import backend as K
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


class ResCNN(object):

	def __init__(self,
				task_type='regression',
				with_skip=True,
				robust=True,
				mae=True,
				inject_Zfrac=True,
				property_name='OQMD_Ef',
				Z_frac_len=9,
				total_elements=103
				):
		self.task_type = task_type
		self.with_skip = with_skip
		self.robust = robust
		self.mae = mae
		self.inject_Zfrac = inject_Zfrac
		self.property_name = property_name
		self.Z_frac_len = Z_frac_len
		self.total_elements = total_elements
		self.efCls = ElementFraction()
		# self.scaler = StandardScaler()
		self.scaler = Normalizer()
		self.tensor_scaler = NormalizeTensor()
		# self.elem_fea = ElemFeatures()
		# self.adaptive_lossfun = (adaptive.AdaptiveLossFunction(
		# 			        num_channels=1, float_dtype=np.float32))



	def build_model(self, input_shape):
		# input_shape = (self.X_train.shape[1], 1)

		inputs = tf.keras.Input(shape=input_shape)
		# Z_frac = tf.keras.Input(shape=Z_frac_shape)


		x1 = layers.Conv1D(64,3, activation='relu', padding='valid')(inputs)
		# x1 = layers.MaxPooling1D(pool_size=(2, ), strides=(2,))(x1)
		x2 = layers.Conv1D(128,3, activation='relu', padding='valid')(x1)
		# x2 = layers.MaxPooling1D(pool_size=(2, ), strides=(2,))(x2)

		# x2 = BatchNormalization()(x2)
		# x3 = layers.Conv1D(256,3, activation='relu')(x2)
		x3 = layers.Conv1D(256,3, activation='relu', padding='valid')(x2)
		# x3 = layers.MaxPooling1D(pool_size=(2, ), strides=(1,))(x3)
		# x3 = BatchNormalization()(x3)

		# x3 = layers.Concatenate()([x3, x_f])
		x = layers.Conv1D(256,3, activation='relu', padding='valid')(x3)
		# x = layers.MaxPooling1D(pool_size=(2, ), strides=(1,))(x)
		# x = layers.Concatenate()([x, x2])
		# x = AttentionWithContext()(x)

		## removed
		# x = layers.Conv1D(256,3, activation='relu', padding='valid')(x)
		

		# x = layers.Conv1D(256,3, activation='relu', padding='valid')(x)
		# x = layers.MaxPooling1D(pool_size=(2, ), strides=(2,))(x)
		# x = BatchNormalization()(x)

		x = layers.GlobalMaxPooling1D()(x)


		x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4), activation='relu')(x)
		x = layers.Dropout(0.2)(x)

		# x = Attention(128)(x)

		x = layers.Dense(1024, kernel_regularizer=regularizers.l2(1e-4), activation='relu')(x)
		x = layers.Dropout(0.2)(x)
		# x = layers.Dense(1024, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x)
		# x = layers.Dropout(0.1)(x)

		## added
		x = layers.Concatenate()([x, layers.Flatten()(x3)])

		x = layers.Dense(1024, kernel_regularizer=regularizers.l2(1e-4), activation='relu')(x)
		x = layers.Dropout(0.2)(x)

		x = layers.Concatenate()([x, layers.Flatten()(x2)])

		## removed
		x = layers.Dense(2048, kernel_regularizer=regularizers.l2(1e-4), activation='relu')(x)
		x = layers.Dropout(0.2)(x)

		# x = layers.Dense(2048, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x)
		# x = layers.Dropout(0.1)(x)

		# x = layers.Dense(2048, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x)
		# x = layers.Dropout(0.1)(x)

		## Removed
		# x = layers.Concatenate()([x, layers.Flatten()(x1)])

		x = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4), activation='relu')(x)
		x = layers.Dropout(0.2)(x)

		out = layers.Dense(1, activation='linear')(x)
		model = tf.keras.Model([inputs], [out], name="ResCNN")
		return model

	def fnet_block(self, inputs):

		# x = layers.LSTM(activation = 'relu' ,units = 100, return_sequences = True)(inputs)
		# x = layers.LSTM(activation = 'relu' ,units = 64, return_sequences = True)(x)
		# x = layers.LSTM(activation = 'relu' ,units = 128, return_sequences = True)(x)
		# x = SeqSelfAttention()(x)
		# x = layers.Attention()(x)
		# out = layers.Flatten()(x)

		x1 = tf.cast(layers.Lambda(tf.signal.rfft)(inputs), dtype=tf.float32)
		x2 = layers.Concatenate()([layers.Flatten()(inputs), layers.Flatten()(x1)])
		x2 = layers.LayerNormalization()(x2)
		x3 = layers.Dense(64, kernel_regularizer=regularizers.l2(1e-6), activation='relu')(x2)
		out = layers.Concatenate()([x2, layers.Flatten()(x3)])
		out = layers.LayerNormalization()(out)
		return out


	def custom_loss(self, y_true, y_pred):
		# loss = pow((y_pred[0] - y_true), 2) / (2 * pow(y_pred[1], 2)) + \
		#        tf.math.log(y_pred[1])
		mean, sigma = tf.split(y_pred, 2, axis=-1)
		# y_true_mean, y_true_std = tf.split(y_true, 2, axis=-1)
		loss =  np.sqrt(2.0) * K.abs(mean - y_true) * K.exp(-sigma) + sigma
		return K.mean(loss)

	def MAE(self, y_true, y_pred):
		# state_dict = self.scaler.state_dict()
		# print('\nst_dict: ', state_dict)
		# self.tensor_scaler.load_state_dict(state_dict)
		mean, sigma = tf.split(y_pred, 2, axis=-1)
		mae = K.abs(self.scaler.denorm(mean) - self.scaler.denorm(y_true))
		# print(K.mean(mae), self.tensor_scaler.denorm(K.mean(mae)))
		return K.mean(mae)
	def denorm_mae(self, y_true, y_pred):
		mae = K.abs(self.scaler.denorm(y_true) - self.scaler.denorm(y_pred))
		# print(K.mean(mae), self.tensor_scaler.denorm(K.mean(mae)))
		return K.mean(mae)



	def parse_formulae(self, data_path=None, data=None):
		if data_path is not None:
			df = pd.read_csv(data_path)
		if data is not None:
			df = data
		df_a = df.apply(lambda x: self.efCls.featurize(Composition(x.formula)), axis=1, result_type='expand')
		df_featurized = pd.concat([df, df_a],axis='columns')
		X = df_featurized.drop(['ID', 'formula', 'integer_formula', 'nsites', 'Z', 'target', 'cif', 'nelements', 'is_inert_gas'], axis=1, errors='ignore')
		y = df_featurized['target']

		return np.expand_dims(X, axis=-1), np.array(y)


	def train(self, train_path=None, val_path=None, data=None, lr=0.0003, batch_size=256, epochs=300, random_state=10):
		print("Loading and processing data ...")
		if (train_path and val_path):
			X_train, y_train = self.parse_formulae(data_path=train_path)
			X_val, y_val = self.parse_formulae(data_path=val_path)

		elif data is not None:
			X_data, y_data = self.parse_formulae(data=data)
			X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.15, random_state=random_state)

		else:
			raise ValueError("Data path or a dataframe containing the data should be provided")
		self.scaler.fit(y_train)
		y_train = self.scaler.norm(y_train)
		y_val = self.scaler.norm(y_val)

		input_shape = (X_train.shape[1], X_train.shape[2])
		model_cnn = self.build_model(input_shape)
		# model_dense = self.build_model(with_cnn=False)
		print(model_cnn.summary())
		if self.mae:
			model_cnn.compile(
					loss='mae',#'mae'
					optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=[self.denorm_mae])
		else:
			model_cnn.compile(
						loss=self.custom_loss,
						optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=[self.MAE])
		# model_dense.compile(
		# 			loss="mae",
		# 			optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=['mae'])
		tf.keras.utils.plot_model(model_cnn, to_file='model_cnn.png')
		# plt.show()
		checkpointer_cnn = tf.keras.callbacks.ModelCheckpoint('saved_models/best_model_ResCNN'+self.property_name+'.h5', monitor='val_denorm_mae', save_best_only=True)#attn_conv was best model 0.0294
		# checkpointer_dense = tf.keras.callbacks.ModelCheckpoint('saved_models/Rex_best_model_dense_clipped.h5', save_best_only=True)
		try:
			# history_cnn = model_cnn.fit([np.expand_dims(np.array(self.X_train), axis=-1), np.expand_dims(np.array(self.X_train_frac), axis=-1)], self.y_train, validation_split=0.15, epochs=epochs, batch_size=batch_size, callbacks=[checkpointer_cnn])
			# history_cnn = model_cnn.fit([np.expand_dims(np.array(self.X_train_wavelet), axis=-1)], self.y_train, validation_split= 0.15, epochs=epochs, batch_size=batch_size, callbacks=[checkpointer_cnn])			
			# history_cnn = model_cnn.fit(np.expand_dims(np.array(self.X_train)[:10000], axis=-1), self.y_train[:10000], validation_data=(np.expand_dims(np.array(self.X_test), axis=-1), self.y_test), epochs=epochs, batch_size=batch_size, callbacks=[checkpointer_cnn])
			history_cnn = model_cnn.fit([X_train], y_train, validation_data=([X_val], y_val), epochs=epochs, batch_size=batch_size, callbacks=[checkpointer_cnn])
			# history_dense = model_dense.fit(np.expand_dims(np.array(self.X_train[list(self.columns)]), axis=-1), self.y_train, validation_split=0.15, epochs=epochs, batch_size=batch_size, callbacks=[checkpointer_dense])
			# with ind
			# history_cnn = model_cnn.fit(self.X_train, self.y_train, validation_split=0.15, epochs=epochs, batch_size=batch_size, callbacks=[checkpointer_cnn])
			# self.plot_training_history(history_cnn, epochs)
			print('############# Testing ###########')
			val_num = train_path.split('/')[-1].split('.')[0][-1]
			if val_num.isdigit():
				# i=val_num
				i=''
			else:
				i=''
			self.test('saved_models/best_model_ResCNN'+self.property_name+'.h5', '/'.join(train_path.split('/')[:-1])+'/test'+i+'.csv')

		except KeyboardInterrupt:
			print('############# Testing ###########')
			val_num = train_path.split('/')[-1].split('.')[0][-1]
			if val_num.isdigit():
				# i=val_num
				i=''
			else:
				i=''
			self.test('saved_models/best_model_ResCNN'+self.property_name+'.h5', '/'.join(train_path.split('/')[:-1])+'/test'+i+'.csv')
		else:
			self.plot_training_history(history_cnn, epochs)


	def plot_training_history(self, history_cnn, epochs):
		fig, ax = plt.subplots(1,1,figsize=(14,6))
		ax.plot(range(1, epochs+1), history_cnn.history['MAE'], c='b', lw='2', ls='-')
		ax.plot(range(1, epochs+1),history_cnn.history['val_MAE'], c='r', lw='2', ls='-')
		ax.set_ylim([0.15, 0.8])
		ax.set_title('With CNN model loss', fontsize=20)
		ax.set_ylabel('MAE', fontsize=18)
		ax.set_xlabel('epoch', fontsize=18)
		ax.legend(['train', 'validation'], loc='upper left', fontsize=18)
		fig.savefig('CNN_training_log.png', dpi=100)
		plt.show()


	def train_ensemble(self, n_members):
		trainX, testX, trainy, testy = train_test_split(self.X_train, self.y_train, test_size=0.1, random_state=10)
		
		
		def train_members(n_members, lr=0.0003, epochs=2):

			for i in range(n_members):
				x_train, x_val, y_train, y_val = train_test_split(trainX, trainy, test_size=0.1, random_state=np.random.choice(20))
				model = self.build_model()
				model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=['mae'])
				model_checkpointer = tf.keras.callbacks.ModelCheckpoint('ensemble_models/best_model_'+str(i)+'_Ef.h5', save_best_only=True)
				history = model.fit(np.expand_dims(np.array(x_train), axis=-1), y_train, validation_data=(np.expand_dims(np.array(x_val), axis=-1), y_val), epochs=epochs, batch_size=64, callbacks=[model_checkpointer])

				# save model
				# filename = 'ensemble_models/model_' + str(i + 1) + '.h5'
				# model.save(filename)
				print('>Saved model ',  (i+1))

		# load models from file
		def load_all_models(self, n_models):
			all_models = list()
			for i in range(n_models):
				# define filename for this ensemble
				filename = 'ensemble_models/best_model_'+str(i)+'_Ef.h5'
				# load model from file
				model = load_model(filename)
				# add to list of members
				all_models.append(model)
				print('>loaded %s' % filename)
			return all_models
		 
		# define stacked model from multiple member input models
		def define_stacked_model(self, members):
			# update all layers in all models to not be trainable
			for i in range(len(members)):
				model = members[i]
				for layer in model.layers:
					# make not trainable
					layer.trainable = False
					# rename to avoid 'unique layer name' issue
					layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
			# define multi-headed input
			ensemble_visible = [model.input for model in members]
			# concatenate merge output from each model
			ensemble_outputs = [model.output for model in members]
			merge = concatenate(ensemble_outputs)
			hidden = Dense(10, activation='relu')(merge)
			output = Dense(3, activation='linear')(hidden)
			model = Model(inputs=ensemble_visible, outputs=output)
			# plot graph of ensemble
			plot_model(model, show_shapes=True, to_file='model_graph.png')
			# compile
			model.compile(loss='mae', optimizer='adam', metrics=['mae'])
			return model
		 
		# fit a stacked model
		def fit_stacked_model(model, inputX, inputy):
			# prepare input data
			X = [inputX for _ in range(len(model.input))]
			# encode output data
			# inputy_enc = to_categorical(inputy)
			# fit model
			model.fit(X, inputy, epochs=300, verbose=0)
		 
		# make a prediction with a stacked model
		def predict_stacked_model(model, inputX):
			# prepare input data
			X = [inputX for _ in range(len(model.input))]
			# make prediction
			return model.predict(X, verbose=0)

		train_members(n_members)
		

	def test(self, model_path, test_path, n_test=100):
		model = tf.keras.models.load_model(model_path, compile=False)
		X_test, y_test = self.parse_formulae(data_path=test_path)

		y_pred = model.predict([X_test])
		# print(y_pred[:,0][:20])
		# print(np.exp(y_pred[:,1][:20]))
		sd = self.scaler.state_dict()
		print(sd)
		if self.mae:
			y_pred = self.scaler.denorm(y_pred)
		else:
			y_pred = self.scaler.denorm(y_pred[:,0])
		# y_test = self.scaler.denorm(y_test)
		mae = mean_absolute_error(y_test, y_pred)
		mape = mean_absolute_percentage_error(y_test, y_pred)
		mse = mean_squared_error(y_test, y_pred)
		r2 = r2_score(y_test, y_pred)

		print('MAE: ', mae)
		print('MAPE: {:.3f}%'.format(mape))
		print('RMSE: ', np.sqrt(mse))
		print('R2 score: ', r2)
		df_preds = pd.DataFrame({'target': y_test.flatten(), 'prediction': y_pred.flatten()})
		df_preds.to_csv("results/test_results_ResCNN.csv", index=False)

		# fig, ax = plt.subplots(1, 1, figsize=(8,8))
		# ax.scatter(self.y_test, y_pred, marker='o', s=4)
		# ax.set_xlabel('Actual Ef')
		# ax.set_ylabel('Predicted Ef')
		# df = pd.DataFrame({'Actual Ef': list(self.y_test), 'Predicted Ef': list(np.array(y_pred).flatten())})
		# df.to_csv('Ef_prediction.csv', sep=',', index=False)
		# plt.show()


		# for i in range(n_test):
		# 	fig, ax = plt.subplots(1,1,figsize=(8,6))
		# 	ax.plot(np.linspace(0, 30, 3000), y_pred[120+i], label='Prediction', c='r', lw=2)
		# 	ax.plot(np.linspace(0, 30, 3000), self.y_test[120+i], label='Ground Truth', c='b', lw=2, ls='--')
		# 	ax.legend(fontsize=18)
		# 	ax.set_xlabel('Energy (eV)', fontsize=18)
		# 	ax.set_ylabel("Real Permittivity", fontsize=18)
		# 	composition = self.X_test.iloc[120+i]['full_formula']
		# 	ax.set_title(composition+ ' dielectric function')
		# 	fig.savefig('test_results_Rex/test_'+str(i)+'.png', dpi=100)
		# 	# plt.show()
		# 	plt.close()

	def transfer_learning(self):
		model = tf.keras.models.load_model('saved_models/best_model_cnn_Ef.h5')
		df = pd.read_csv('data/datasets/MP_Ed_elem_frac.csv', sep=',')
		X_Ed = df.drop(['composition', 'Ed'], axis=1)
		y = df['Ed']
		# print(model.layers)
		for l in model.layers:
			if not (l.name=='dense_5'):
				l.trainable = False
				# print(l.name)
		X_train, X_test, y_train, y_test = train_test_split(X_Ed, y, test_size=0.1, random_state=10)

		checkpointer_TL = tf.keras.callbacks.ModelCheckpoint('saved_models/best_model_cnn_wavelet_Ef.h5', save_best_only=True)
		model.fit(np.expand_dims(np.array(X_train), axis=-1), y_train, validation_split=0.15, epochs=300, batch_size=32, callbacks=[checkpointer_TL])

	def test_TL(self):
		model = tf.keras.models.load_model('saved_models/best_model_cnn_Ed_TL.h5', compile=False)
		df = pd.read_csv('data/datasets/MP_Ed_elem_frac.csv', sep=',')
		X_Ed = df.drop(['composition', 'Ed'], axis=1)
		y = df['Ed']
		X_train, X_test, y_train, y_test = train_test_split(X_Ed, y, test_size=0.1, random_state=10)

		y_pred = model.predict(np.expand_dims(np.array(X_test), axis=-1))
		mae = mean_absolute_error(y_test, y_pred)
		mape = mean_absolute_percentage_error(y_test, y_pred)
		mse = mean_squared_error(y_test, y_pred)
		r2 = r2_score(y_test, y_pred)

		print('MAE: ', mae)
		print('MAPE: {:.3f}%'.format(mape))
		print('RMSE: ', np.sqrt(mse))
		print('R2 score: ', r2)

		fig, ax = plt.subplots(1, 1, figsize=(8,8))
		ax.scatter(y_test, y_pred, marker='o', s=4)
		ax.set_xlabel('Actual Ed')
		ax.set_ylabel('Predicted Ed')
		plt.show()

	def visualize_filters(self, model):
		for l in model.layers:
			print(l.name)
		model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[5].output)
		test_data = np.expand_dims(np.array(self.X_test), axis=-1)
		test_input = test_data[12345].reshape((1, test_data.shape[1], test_data.shape[2]))
		conv1_maps = model.predict(test_input)#.reshape(1, 101, 1, 64)
		print(conv1_maps[:,:,1])
		# print(model.summary())
		for i in range(256):
			fmap = conv1_maps[:,:,i]
			print(np.count_nonzero(fmap))
			# plt.imshow(fmap.reshape(93,1))
			# plt.xticks([])
			# plt.yticks([])
			# plt.show()
			# plt.close()
		# width = 8
		# height = 8
		# ix = 1
		# for i in range(width):
		#   for j in range(height):
		#     # specify subplot and turn of axis
		#     ax = plt.subplot(width, height, ix)
		#     ax.set_xticks([])
		#     ax.set_yticks([])
		#     # plot filter channel in grayscale
		#     ax.imshow(conv1_maps[0, :, :, ix-1])
		#     ix += 1
		# plt.show()




if __name__=='__main__':

	cnn = ResCNN()
	cnn.train(train_path='data/databases/difunc_data/RealX_train.pkl', val_path='data/databases/difunc_data/RealX_val.pkl', epochs=800)
	# cnn.train(train_path='data/databases/MP/shear_modulus_vrh/train.csv', val_path='data/databases/MP/shear_modulus_vrh/val.csv', epochs=800)
	# cnn.train(train_path='../../CrabNet/data/benchmark_data/OQMD_Formation_Enthalpy/train.csv', val_path='../../CrabNet/data/benchmark_data/OQMD_Formation_Enthalpy/val.csv', epochs=800)
	# cnn.test('saved_models/best_model_cnn_OQMD_Ef.h5', '../../CrabNet/data/benchmark_data/OQMD_Formation_Enthalpy/test.csv')
	# cnn.train(train_path='../../CrabNet/data/benchmark_data/OQMD_Formation_Enthalpy/train.csv', val_path='../../CrabNet/data/benchmark_data/OQMD_Formation_Enthalpy/val.csv', epochs=800)
	# cnn.train(train_path='../../CrabNet/data/matbench_cv/mp_e_form/train0.csv', val_path='../../CrabNet/data/matbench_cv/mp_e_form/val0.csv', epochs=800)
