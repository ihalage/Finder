import numpy as np
import pandas as pd
import json
import warnings
import sys

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import argparse
import logging
import os

"""Silence every warning of notice from tensorflow."""
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError, Huber
import tensorflow.keras.backend as K

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from utils import RobustMAE, RobustLoss
from utils import Normalizer, NormalizeTensor

from spektral.data import DisjointLoader
from data_loader import DataLoader
from model import Net

import matplotlib.pyplot as plt
from matplotlib import rcParams



from pymatgen import Composition, Element, MPRester
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from matminer.featurizers.composition import ElementFraction


# reproducibility
# np.random.seed(1)
# tf.random.set_seed(2)
# import random as python_random
# np.random.seed(3)
# python_random.seed(213)
# tf.random.set_seed(None)	# 2-0.277, 123-settles at 0.265, 17-0.253 best-42 - 0.246


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

# scaler = Normalizer()

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.63 #0.63
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

class Worker(object):
	def __init__(self,
				train_path,
				val_path,
				test_path,
				model_path='saved_models/best_model_gnn',
				learning_rate= 3e-4,
				epochs=500,
				batch_size=128,
				patience=100,
				num_targets=1,
				channels=200,
				aggregate_type='mean',
				mae_loss=False,
				train=False,
				test=False,
				pred_func=False,
				is_pickle=False,
				use_edge_predictor=False,
				use_crystal_structure=False,
				embedding_path='data/embeddings/',
				embedding_type='mat2vec',
				task_type='regression',
				max_no_atoms=500):

		self.train_path = train_path
		self.val_path = val_path
		self.test_path = test_path
		self.model_path = model_path
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.batch_size = batch_size
		self.patience = patience
		self.num_targets = num_targets
		self.channels = channels
		self.aggregate_type = aggregate_type
		self.mae_loss = mae_loss
		self.train = train
		self.test = test
		self.pred_func = pred_func
		self.is_pickle = is_pickle
		self.use_edge_predictor = use_edge_predictor
		self.use_crystal_structure = use_crystal_structure
		self.embedding_path = embedding_path
		self.embedding_type = embedding_type
		self.task_type = task_type
		self.max_no_atoms = max_no_atoms

		self.scaler = Normalizer()

	def train_model(self):
		dataset_tr = DataLoader(data_path=self.train_path, 
								is_train=self.train,
								pred_func=self.pred_func,
								is_pickle=self.is_pickle,
								scaler=self.scaler,
								embedding_path=self.embedding_path,
								embedding_type=self.embedding_type,
								use_edge_predictor=self.use_edge_predictor,
								use_crystal_structure=self.use_crystal_structure,
								task_type=self.task_type)

		## get scaler attribute after fitting on training data
		self.scaler = dataset_tr.scaler
		scaler_dict = self.scaler.state_dict()
		os.makedirs('saved_models/best_model_gnn', exist_ok=True)
		json.dump(scaler_dict, open("saved_models/best_model_gnn/scaler_dict.json", 'w' ))	# save the state of scaler

		# print(self.scaler.mean, self.scaler.std)
		
		dataset_val = DataLoader(data_path=self.val_path, 
								is_train=False,
								pred_func=self.pred_func,
								is_pickle=self.is_pickle,
								scaler=self.scaler,
								embedding_path=self.embedding_path,
								embedding_type=self.embedding_type,
								use_crystal_structure=self.use_crystal_structure,
								task_type=self.task_type)

		loader_tr = DisjointLoader(dataset_tr, batch_size=self.batch_size, epochs=self.epochs)
		loader_val = DisjointLoader(dataset_val, batch_size=self.batch_size, epochs=1)

		# 	# Parameters
		# F = dataset_tr.n_node_features  # Dimension of node features
		# N = dataset_tr.n_nodes
		# S = dataset_tr.n_edge_features  # Dimension of edge features
		n_out = dataset_tr.n_labels  # Dimension of the target
		if self.mae_loss:
			n_out/=2			

		model = Net(channels=self.channels,
				n_out=n_out,#self.num_targets
				aggregate_type=self.aggregate_type,
				use_edge_predictor=self.use_edge_predictor,
				use_crystal_structure=self.use_crystal_structure)


		optimizer = Adam(self.learning_rate)
		if self.mae_loss:
			loss_fn = MeanAbsoluteError()
		else:
			loss_fn = RobustLoss()
		robust_mae = RobustMAE(scaler=self.scaler, pred_func=self.pred_func)

		try:
			if self.train:
				################################################################################
				# Fit model
				################################################################################
				@tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
				def train_step(inputs, target):
					with tf.GradientTape() as tape:
						predictions = model(inputs, training=True)
						target=tf.cast(target, tf.float32)
						# l = K.print_tensor(predictions, message='predictions = ')
						# l = K.print_tensor(loss_fn(target, predictions), message='Loss = ')
						# l = K.print_tensor(sum(model.losses), message='model loss = ')
						loss = loss_fn(target, predictions) + sum(model.losses)
						# weights = model.weights
						# l = K.print_tensor(weights, message='weights = ')#/////////////////////remove
						if self.mae_loss:
							mae = loss_fn(target, predictions)
						else:
							mae = robust_mae.mean_absolute_error(target, predictions)
					gradients = tape.gradient(loss, model.trainable_variables)
					# l = K.print_tensor(gradients, message='gradients = ')
					# gdnorm = tf.linalg.global_norm(gradients)
					# l = K.print_tensor(gdnorm, message='gdnorm = ')
					# print(gradients)
					# gradients, _ = tf.clip_by_global_norm(gradients, 1.0)#///////////////////////////////////
					# l = K.print_tensor(gradients, message='new gradients = ')
					# tf.debugging.check_numerics(gradients, "gradients have nan!")
					optimizer.apply_gradients(zip(gradients, model.trainable_variables))

					return loss, mae

				train_mae = []
				validation_mae = []
				step = loss = 0
				tr_mae = 0
				# val_mae = 0
				validation_data = list(loader_val)
				epoch_no = 1
				best_val_mae = 1e6
				for batch in loader_tr:
					step += 1
					# loss += train_step(*batch)
					l, tmae = train_step(*batch)
					loss+=l
					tr_mae+=tmae
					if step == loader_tr.steps_per_epoch:
						val_loss = 0
						val_mae = 0
						# acc=0
						# loader_val = DisjointLoader(dataset_val, batch_size=batch_size, epochs=1)
						for batch_val in validation_data:
							val_inputs, val_targets = batch_val
							val_predictions = model(val_inputs, training=False)
							val_loss += loss_fn(val_targets, val_predictions)
							# acc += tf.reduce_mean(categorical_accuracy(val_targets, val_predictions))
							if self.mae_loss:
								val_mae+=loss_fn(val_targets, val_predictions)
							else:
								val_mae += robust_mae.mean_absolute_error(val_targets, val_predictions)
							# val_loss_total+=val_loss
						step = 0
						K.set_value(optimizer.learning_rate, optimizer.lr*0.999)
						# print(K.eval(optimizer.lr))
						print('\nEpoch: ', epoch_no)
						print("Training Loss: {} \t Validation Loss: {}\n".format(loss / loader_tr.steps_per_epoch, val_loss / loader_val.steps_per_epoch))
						print("Training MAE: {} \t Validation MAE: {}\n".format(tr_mae / loader_tr.steps_per_epoch, val_mae / loader_val.steps_per_epoch))

						train_mae.append(tr_mae/loader_tr.steps_per_epoch)
						validation_mae.append(val_mae/loader_val.steps_per_epoch)

						if val_mae/loader_val.steps_per_epoch < best_val_mae:
							# save current best model and scaler metadata
							model.save('saved_models/best_model_gnn',save_format='tf')
							# model_json = model.to_json()
							

						if len(validation_mae) > self.patience:
							if validation_mae[-(self.patience+1)] < min(validation_mae[-self.patience:]):
								print(f'\nEarly stopping. No validation loss '
		                      			f'improvement in {self.patience} epochs.')
								break

						with open('results/history.csv', 'a+') as file:
							file.write(str(epoch_no)+','+str((tr_mae/loader_tr.steps_per_epoch).numpy())+','+str((val_mae/loader_val.steps_per_epoch).numpy())+'\n')

						# print(train_mae)
						# print(validation_mae)
						# if len(train_mae)==5:
						# 	plt.plot(train_mae)
						# 	plt.show()
						# print('validation accuracy: {}'.format(acc/loader_val.steps_per_epoch))
						epoch_no+=1
						# print("Validation Loss: {}".format(val_loss / loader_val.steps_per_epoch))
						loss = 0
						tr_mae = 0
				tm = [t.numpy() for t in train_mae]
				vm = [v.numpy() for v in validation_mae]
				df = pd.DataFrame({'Train MAE': tm, 'Validation MAE': vm})
				df.to_csv('results/training_history.csv')
				##################
				## plotting
				plt.plot(range(1, len(train_mae)+1), train_mae, lw=2, ls='-', c='blue', label='Train')
				plt.plot(range(1, len(validation_mae)+1), validation_mae, lw=2, ls='-', c='red', label='Validation')
				plt.xlabel('Epoch Number', fontsize=14)
				plt.ylabel('Mean Absolute Error', fontsize=14)
				plt.legend()
				plt.tight_layout()
				plt.savefig('results/training_log.png', dpi=100)
				plt.show()

		except KeyboardInterrupt:
			pass

		if self.test:
			################################################################################
			# Evaluate model
			################################################################################

			self.test_model()

			# print("\n\nLoading current best model ...\n")
			# try:
			# 	model_path = self.model_path
			# 	model = tf.keras.models.load_model(model_path)
			# 	# model = tf.keras.models.load_model('saved_models/best_model_gnn')
			# 	# model_data = json.load(open("saved_models/best_model_gnn.json"))
			# except:
			# 	print('No model exists. Please run with --train to train the model first')

			# # model = keras.models.model_from_json(model_data['model'])
			# scaler_dict = json.load(open("{0}/scaler_dict.json".format(self.model_path)))
			# # scaler_dict = json.load(open("saved_models/best_model_gnn/scaler_dict.json"))
			# self.scaler.load_state_dict(state_dict=scaler_dict)
			# robust_mae = RobustMAE(scaler=self.scaler, pred_func=self.pred_func) # update scaler 

			# dataset_te = DataLoader(data_path=self.test_path, 
			# 					is_train=False,
			# 					pred_func=self.pred_func,
			# 					is_pickle=self.is_pickle,
			# 					scaler=self.scaler,
			# 					embedding_path=self.embedding_path,
			# 					embedding_type=self.embedding_type,
			# 					use_crystal_structure=self.use_crystal_structure,
			# 					task_type=self.task_type)
			# loader_te = DisjointLoader(dataset_te, batch_size=self.batch_size, epochs=1, shuffle=False)

			# print('Testing the model ...\n')
			# loss = 0
			# test_mae = 0
			# target_list = []
			# predictions_list = []
			# uncertainity_list = []
			# for batch in loader_te:
			# 	inputs, target = batch
			# 	predictions = model(inputs, training=False)
			# 	# preds = model.predict(inputs)
			# 	# print(preds.shape)
			# 	# print(predictions.numpy(), target.numpy())
			# 	# target_list.append(tf.split(target, 2, axis=-1)[0])
				
			# 	loss += loss_fn(target, predictions)
			# 	if self.mae_loss:
			# 		predictions_list.extend(list(predictions.numpy()))
			# 		test_mae+=loss_fn(target, predictions)
			# 	else:
			# 		predictions_list.extend(list(tf.split(predictions, 2, axis=-1)[0].numpy()))
			# 		uncertainity_list.extend(list(tf.split(predictions, 2, axis=-1)[1].numpy()))
			# 		test_mae += robust_mae.mean_absolute_error(target, predictions)
			# loss /= loader_te.steps_per_epoch
			# print("Test Loss: {}".format(loss / loader_te.steps_per_epoch))
			# print("Test MAE denormed: {}".format(test_mae / loader_te.steps_per_epoch))

			# if self.pred_func:
			# 	# print(predictions_list[20][-10:])
			# 	# print(len(predictions_list), len(predictions_list[10]))
			# 	# preds = model.predict(loader_te.load())
			# 	predictions = np.array(predictions_list)
			# 	# uncertainities = np.array(uncertainity_list)
			# 	target_df = pd.read_pickle(self.test_path)
			# 	targets = np.array(target_df.iloc[:,2:])
			# 	for i in range(10, 30):
			# 		plt.plot(predictions[i], label='Prediction')
			# 		plt.plot(targets[i], label='Ground truth')
			# 		plt.title(target_df.iloc[i]['formula'])
			# 		plt.legend()
			# 		plt.show()
			# 	r2 = r2_score(predictions, targets)
			# 	mse = mean_squared_error(predictions, targets)
			# 	print("Test R2 Score: {}".format(r2))
			# 	print("Test RMSE: {}".format(np.sqrt(mse)))
			# 	# print(preds)
			# 	df_preds = pd.DataFrame({'target': targets, 'prediction': predictions})
			# 	# df_preds.to_csv("results/test_results.csv", index=False)
			# else:
			# 	preds = np.array(predictions_list).flatten()
			# 	sigmas = np.array(uncertainity_list).flatten()
			# 	predictions = self.scaler.denorm(preds)
			# 	uncertainities = tf.math.exp(sigmas) * self.scaler.std
			# 	targets = np.array(pd.read_csv(self.test_path)['target'])
			# 	mae = mean_absolute_error(predictions, targets)
			# 	r2 = r2_score(predictions, targets)
			# 	mse = mean_squared_error(predictions, targets)
			# 	print("Test MAE: {}".format(mae))
			# 	print("Test R2 Score: {}".format(r2))
			# 	print("Test RMSE: {}".format(np.sqrt(mse)))
			# 	# print(preds)
			# 	df_preds = pd.DataFrame({'target': targets, 'prediction': predictions, 'uncertainity': uncertainities})
			# 	df_preds.to_csv("results/test_results.csv", index=False)


	def test_model(self):
		################################################################################
		# Evaluate model
		################################################################################

		print("\n\nLoading current best model ...\n")
		try:
			model_path = self.model_path
			model = tf.keras.models.load_model(model_path)
			# model = tf.keras.models.load_model('saved_models/best_model_gnn')
			# model_data = json.load(open("saved_models/best_model_gnn.json"))
		except:
			print('No model exists. Please run with --train to train the model first')


		if self.mae_loss:
			loss_fn = MeanAbsoluteError()
		else:
			loss_fn = RobustLoss()
		robust_mae = RobustMAE(scaler=self.scaler, pred_func=self.pred_func)

		# model = keras.models.model_from_json(model_data['model'])
		scaler_dict = json.load(open("{0}/scaler_dict.json".format(self.model_path)))
		# scaler_dict = json.load(open("saved_models/best_model_gnn/scaler_dict.json"))
		self.scaler.load_state_dict(state_dict=scaler_dict)
		robust_mae = RobustMAE(scaler=self.scaler, pred_func=self.pred_func) # update scaler 

		dataset_te = DataLoader(data_path=self.test_path, 
							is_train=False,
							pred_func=self.pred_func,
							is_pickle=self.is_pickle,
							scaler=self.scaler,
							embedding_path=self.embedding_path,
							embedding_type=self.embedding_type,
							use_crystal_structure=self.use_crystal_structure,
							task_type=self.task_type)
		loader_te = DisjointLoader(dataset_te, batch_size=self.batch_size, epochs=1, shuffle=False)

		print('Testing the model ...\n')
		loss = 0
		test_mae = 0
		target_list = []
		predictions_list = []
		uncertainity_list = []
		for batch in loader_te:
			inputs, target = batch
			predictions = model(inputs, training=False)
			# preds = model.predict(inputs)
			# print(preds.shape)
			# print(predictions.numpy(), target.numpy())
			# target_list.append(tf.split(target, 2, axis=-1)[0])
			
			loss += loss_fn(target, predictions)
			if self.mae_loss:
				predictions_list.extend(list(predictions.numpy()))
				# test_mae+=loss_fn(target, predictions)
			else:
				predictions_list.extend(list(tf.split(predictions, 2, axis=-1)[0].numpy()))
				uncertainity_list.extend(list(tf.split(predictions, 2, axis=-1)[1].numpy()))
				test_mae += robust_mae.mean_absolute_error(target, predictions)
		loss /= loader_te.steps_per_epoch
		print("Test Loss: {}".format(loss / loader_te.steps_per_epoch))
		print("Test MAE denormed: {}".format(test_mae / loader_te.steps_per_epoch))

		if self.pred_func:
			# print(predictions_list[20][-10:])
			# print(len(predictions_list), len(predictions_list[10]))
			# preds = model.predict(loader_te.load())
			def vectorize(df):
				ef = ElementFraction()
				a_df = df.apply(lambda x: ef.featurize(Composition(x.formula)), axis=1, result_type='expand')
				if (set(['formula', 'formation_energy_per_atom', 'e_above_hull', 'red_formula']).issubset(df.columns.tolist())):
					df_vec = pd.concat([df[['formula', 'formation_energy_per_atom', 'e_above_hull', 'red_formula']], a_df], axis='columns')
				else:
					df_vec = pd.concat([df[['formula']], a_df], axis='columns')

				return df_vec

			
			'''
			model = tf.keras.models.load_model('../../../../saved_models/{0}_best_model_skipcnn_clipped.h5'.format('Imx'), compile=False)
			# x = pd.read_pickle(self.test_path)
			
			# x = vectorize(x)
			# print(x)
			# x = x.drop(['formula'], axis=1)
			# x = np.array(x)

			# rescnn_predictions = model.predict(x)

			rescnn_predictions = np.load('data/databases/difunc_data/predictions/y_pred_Imx.npy')
			predictions = np.array(predictions_list)
			# uncertainities = np.array(uncertainity_list)
			target_df = pd.read_pickle(self.test_path)
			targets = np.array(target_df.iloc[:,2:])
			for i in range(targets.shape[0]):
				plt.plot(np.linspace(0, 30, 3000), predictions[i], label='Finder', c='r', lw=2)
				plt.plot(np.linspace(0, 30, 3000), rescnn_predictions[i], label='ResCNN', c='b', lw=2)
				plt.plot(np.linspace(0, 30, 3000), targets[i], label='Ground truth', c='g', lw=2, ls='--')
				plt.title(target_df.iloc[i]['formula'], fontsize=20)
				plt.xlabel('Energy (eV)', fontsize=18)
				plt.ylabel("Imaginary permittivity", fontsize=18)
				if target_df.iloc[i]['formula'] == 'Ba4LiCu(CO5)2':
					plt.legend(fontsize=14.9)
				else:
					plt.legend(fontsize=16)
				plt.savefig('ImagX_results/{0}_{1}.png'.format(str(i), target_df.iloc[i]['formula']), dpi=100)
				plt.close()

			r2 = r2_score(predictions, targets)
			mse = mean_squared_error(predictions, targets)
			mae = mean_absolute_error(predictions, targets)
			mad = np.mean(np.absolute(targets - np.mean(targets)))

			print('MAE: ', mae)
			print("Test R2 Score: {}".format(r2))
			print("Test RMSE: {}".format(np.sqrt(mse)))
			print('MAD/MAE: {:.3f}'.format(mad/mae))
			# print(preds)
			df_preds = pd.DataFrame({'target': targets, 'prediction': predictions})
			# df_preds.to_csv("results/test_results.csv", index=False)
			


			'''
			df = pd.read_csv('data/databases/difunc_data/MP_no_jarvis.csv')#.iloc[:85790]
			y_pred_real = np.load('RealX_predictions_MP_no_jarvis.npy')
			y_pred_imag = np.load('ImagX_predictions_MP_no_jarvis.npy')
			# predictions = np.array(predictions_list)
			# print(predictions.shape)
			# np.save('ImagX_predictions_MP_no_jarvis.npy', predictions)
			# for i in range(10):
			# 	plt.plot(np.linspace(0, 30, 3000), predictions[i], label='Finder', c='r', lw=2)
			# 	plt.show()

			plasma_freq_list = []
			imag_plasma_list = []
			for i in range(df.shape[0]):
				print(i)
				plasma_freq = 0
				imag_plasma = 0
				realxx = y_pred_real[i]
				imagxx = y_pred_imag[i]
				for j in range(len(y_pred_real[0])):
					if j>10 and  realxx[j-1]>0 and realxx[j]<0:
						plasma_freq = 30.0*j/len(y_pred_real[0]) #if row.mbj_en[j]<=30 else np.nan
						imag_plasma = imagxx[j] #if plasma_freq<=30 else np.nan
						break
				plasma_freq_list.append(plasma_freq)
				imag_plasma_list.append(imag_plasma)

			df['crossover_energy'] = plasma_freq_list
			df['imaginary_permittivity'] = imag_plasma_list
			df_sorted = df.sort_values(by=['crossover_energy'])
			df_sorted = df_sorted[['formula', 'imaginary_permittivity', 'crossover_energy', \
			'e_above_hull', 'formation_energy_per_atom', 'in_ICSD']]
			df_sorted.to_csv('MP_predictions_no_jarvis_Finder.csv', sep=',', index=False)
			
			df_promising = df_sorted[(df_sorted.crossover_energy>0.5) & (df_sorted.crossover_energy<12.4) & (df_sorted.crossover_energy>0.5) & (df_sorted.imaginary_permittivity<2) & (df_sorted.e_above_hull<0.025)]
			df_promising.to_csv('Promising_ENZ_Finder.csv', sep=',', index=False)


		else:
			preds = np.array(predictions_list).flatten()
			sigmas = np.array(uncertainity_list).flatten()
			predictions = self.scaler.denorm(preds)
			uncertainities = tf.math.exp(sigmas) * self.scaler.std
			targets = np.array(pd.read_csv(self.test_path)['target'])
			mae = mean_absolute_error(predictions, targets)
			r2 = r2_score(predictions, targets)
			mse = mean_squared_error(predictions, targets)
			print("Test MAE: {}".format(mae))
			print("Test R2 Score: {}".format(r2))
			print("Test RMSE: {}".format(np.sqrt(mse)))
			# print(preds)
			df_preds = pd.DataFrame({'target': targets, 'prediction': predictions, 'uncertainity': uncertainities})
			df_preds.to_csv("results/test_results.csv", index=False)



def argument_parser():
	'''
	parse input arguments
	'''
	parser = argparse.ArgumentParser(
	 description=("Materials property prediction with CrystalNet")
	 )

	parser.add_argument(
		'--train-path',
		type=str,
		default='data/databases/MP/formation_energy/train.csv',
		help='Path to the training database'
		)
	parser.add_argument(
		'--val-path',
		type=str,
		default='data/databases/MP/formation_energy/val.csv',
		help='Path to the validation database'
		)
	parser.add_argument(
		'--test-path',
		type=str,
		default='data/databases/MP/formation_energy/test.csv',
		help='Path to the test database'
		)
	parser.add_argument(
		'--model-path',
		type=str,
		default='saved_models/best_model_gnn',
		help='Path to the saved model'
		)
	parser.add_argument(
		'--learning-rate',
		type=float,
		default=3e-4,
		help='Learning rate'
		)
	parser.add_argument(
		'--epochs',
		type=int,
		default=500,
		help='Number of epochs'
		)
	parser.add_argument(
		'--batch-size',
		type=int,
		default=128,
		help='Batch size'
		)
	parser.add_argument(
		'--patience',
		type=int,
		default=300,
		help='Patience for early stopping'
		)
	parser.add_argument(
		'--num-targets',
		type=int,
		default=1,
		help='Number of units in the output layer of the network'
		)
	parser.add_argument(
		'--channels',
		type=int,
		default=200,
		help='Dimension of the internal node representation of the graph layer'
		)
	parser.add_argument(
		'--aggregate-type',
		type=str,
		default='mean',
		help='A permutationally invariant function to aggregate messages (e.g. mean, sum, min, max)'
		)
	parser.add_argument(
		'--mae-loss', 
		help='Use mean absolte error as the loss function instead of default Robust loss', 
		action='store_true')
	parser.add_argument(
		'--train', 
		dest='train',
		help='Specify whether to train the network', 
		action='store_true')
	# parser.add_argument(
	# 	'--validate', 
	# 	dest='validate',
	# 	help='Specify whether to validate during training', 
	# 	action='store_true')
	parser.add_argument(
		'--test', 
		dest='test',
		help='Specify whether to test the network', 
		action='store_true')
	parser.add_argument(
		'--pred-func', 
		help='Predict a function instead of a single target at the output layer (multi-target output)', 
		action='store_true')
	parser.add_argument(
		'--is-pickle', 
		help='If data is stored in a pickle file instead of csv file', 
		action='store_true')
	parser.add_argument(
		'--use-edge-predictor', 
		# dest='test',
		help='''Get the distance matrix for crystals available in training set and optimise the loss of 
			predicting edges (in addition to model loss) during training. Does not apply for validation and testing. 
			This requires a column 'cif' in train.csv file containing the cif of all crystals. 
			May take a few minutes to process ''', 
		action='store_true')
	parser.add_argument(
		'--use-crystal-structure', 
		# dest='test',
		help='''Use the crystal structures to learn material properties. 
			This requires a column 'cif' in train.csv, val.csv and test.csv files containing the cif of all crystals. 
			May take a few minutes to process ''', 
		action='store_true')
	parser.add_argument(
		'--embedding-path',
		type=str, 
		default='data/embeddings/',
		help='Path where the element embedding JSON files are saved', 
		)
	parser.add_argument(
		'--embedding-type',
		type=str, 
		default='mat2vec',
		help='Element embedding type. Available embedding types: mat2vec, onehot, cgcnn, megnet16', 
		)
	parser.add_argument(
		'--task-type',
		type=str, 
		default='regression',
		help='Specify whether this is a regression or classification task', 
		)
	parser.add_argument(
		'--max-no-atoms',
		type=int,
		default=500,
		help='Maximum number of atoms that can be in the integer formula (E.g. BaTiO3 has 5 atoms)'
		)

	args = parser.parse_args(sys.argv[1:])
	return args



if __name__=='__main__':

	args = argument_parser()
	worker = Worker(**vars(args))
	if args.train:
		worker.train_model()
	elif args.test:
		worker.test_model()

	# train_path='data/databases/MP/formation_energy/train.csv'
	# val_path = 'data/databases/MP/formation_energy/test.csv'
	# # train_path='data/databases/MP/dielectric_constant/train.csv'
	# # val_path = 'data/databases/MP/dielectric_constant/val.csv'
	# dataset_tr = DataLoader(train_path, is_train=True)
	# dataset_val = DataLoader(val_path, is_train=False)
	# # dataset_te = DataLoader('data/databases/MP_formation_energy/test.csv')
	# learning_rate = 3e-4  # Learning rate
	# epochs = 500  # Number of training epochs
	# batch_size = 128  # Batch size

	# # Parameters
	# F = dataset_tr.n_node_features  # Dimension of node features
	# N = dataset_tr.n_nodes
	# S = dataset_tr.n_edge_features  # Dimension of edge features
	# n_out = dataset_tr.n_labels  # Dimension of the target

	# loader_tr = DisjointLoader(dataset_tr, batch_size=batch_size, epochs=epochs)
	# loader_val = DisjointLoader(dataset_val, batch_size=batch_size, epochs=1)
	# # loader_te = DisjointLoader(dataset_te, batch_size=batch_size, epochs=1, shuffle=False)

	# model = Net()
	# optimizer = Adam(learning_rate)
	# loss_fn = RobustLoss()
	# robust_mae = RobustMAE(scaler=scaler)

	# ################################################################################
	# # Fit model
	# ################################################################################
	# @tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
	# def train_step(inputs, target):
	# 	with tf.GradientTape() as tape:
	# 		predictions = model(inputs, training=True)
	# 		target=tf.cast(target, tf.float32)
	# 		loss = loss_fn(target, predictions) + sum(model.losses)
	# 		mae = robust_mae.mean_absolute_error(target, predictions)
	# 	gradients = tape.gradient(loss, model.trainable_variables)
	# 	optimizer.apply_gradients(zip(gradients, model.trainable_variables))
	# 	return loss, mae

	# train_mae = []
	# validation_mae = []
	# step = loss = 0
	# tr_mae = 0
	# # val_mae = 0
	# validation_data = list(loader_val)
	# epoch_no = 1
	# best_val_loss = 100
	# for batch in loader_tr:
	# 	step += 1
	# 	# loss += train_step(*batch)
	# 	l, tmae = train_step(*batch)
	# 	loss+=l
	# 	tr_mae+=tmae
	# 	if step == loader_tr.steps_per_epoch:
	# 		val_loss = 0
	# 		val_mae = 0
	# 		# acc=0
	# 		# loader_val = DisjointLoader(dataset_val, batch_size=batch_size, epochs=1)
	# 		for batch_val in validation_data:
	# 			val_inputs, val_targets = batch_val
	# 			val_predictions = model(val_inputs, training=False)
	# 			val_loss += loss_fn(val_targets, val_predictions)
	# 			# acc += tf.reduce_mean(categorical_accuracy(val_targets, val_predictions))
	# 			val_mae += robust_mae.mean_absolute_error(val_targets, val_predictions)
	# 			# val_loss_total+=val_loss
	# 		step = 0
	# 		print('\nEpoch: ', epoch_no)
	# 		print("Training Loss: {} \t\t\t Validation loss: {}\n".format(loss / loader_tr.steps_per_epoch, val_loss / loader_val.steps_per_epoch))
	# 		print("Training MAE: {} \t\t\t Validation MAE: {}\n".format(tr_mae / loader_tr.steps_per_epoch, val_mae / loader_val.steps_per_epoch))

	# 		train_mae.append(tr_mae/loader_tr.steps_per_epoch)
	# 		validation_mae.append(val_mae/loader_val.steps_per_epoch)

	# 		# print(train_mae)
	# 		# print(validation_mae)
	# 		# if len(train_mae)==5:
	# 		# 	plt.plot(train_mae)
	# 		# 	plt.show()
	# 		# print('validation accuracy: {}'.format(acc/loader_val.steps_per_epoch))
	# 		epoch_no+=1
	# 		# print("Validation Loss: {}".format(val_loss / loader_val.steps_per_epoch))
	# 		loss = 0
	# 		tr_mae = 0

	# ##################
	# ## plotting
	# plt.plot(range(1, epochs+1), train_mae, lw=2, ls='-', c='blue', label='Train')
	# plt.plot(range(1, epochs+1), validation_mae, lw=2, ls='-', c='red', label='Validation')
	# plt.xlabel('Epoch Number', fontsize=14)
	# plt.ylabel('Mean Absolute Error', fontsize=14)
	# plt.legend()
	# plt.tight_layout()
	# plt.savefig('energy_per_atom.png', dpi=100)
	# plt.show()


	# ################################################################################
	# # Evaluate model
	# ################################################################################
	# print("Testing model")
	# loss = 0
	# target_list = []
	# predictions_list = []
	# for batch in loader_te:
	# 	inputs, target = batch
	# 	predictions = model(inputs, training=False)
	# 	# preds = model.predict(inputs)
	# 	# print(preds.shape)
	# 	# print(predictions.numpy(), target.numpy())
	# 	loss += loss_fn(target, predictions)
	# loss /= loader_te.steps_per_epoch
	# print("Done. Test loss: {}".format(loss))
	# loader_te = DisjointLoader(dataset_te, batch_size=batch_size, epochs=1, shuffle=False)
	# preds = model.predict(loader_te.load())
	# # print(preds)
	# df_preds = pd.DataFrame({"predictions": preds.flatten()})
	# df_preds.to_csv("OQMD__Eform_preds_MPCNN.csv", header=None, index=False)

	# print(preds.shape)


