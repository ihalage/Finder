import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError
import tensorflow.keras.backend as K

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from utils import RobustMAE, RobustLoss
from utils import Normalizer
from model import Finder

from spektral.data import DisjointLoader
from data_loader import DataLoader

from pymatgen import Composition, Element, MPRester
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from matminer.featurizers.composition import ElementFraction


class Worker(object):
	"""
	**Arguments**
	- `train_path`: path to the training database
	- `val_path`: path to the validation database
	- `test_path`: path to the test database
	- `model_path`: path where the best model is saved
	- `epochs`: number of epochs
	- `learning_rate`: initial learning rate, which is lowered at every epoch by a factor of 0.999
	- `batch_size`: minibatch size
	- `patience`: number of epochs to wait with no improvement in loss before stopped by early stopping criterion 
	- `channels`: internal vector dimension of the message passing layer;
	- `aggregate_type`: permutation invariant function to aggregate messages
	- `mae_loss`: use mean absolute error as the loss function instead of default L1 robust loss
	- `train`: flag to train Finder model
	- `test`: flag to test Finder model
	- `pred_func`: predict a function (multi-target regression)
	- `is_pickle`: use this flag if data is stored as a pickle file
	- `threshold_radius`: atoms located at a distance less than the threshold radius are bonded in the crystal graph
	- `use_crystal_structure`: use crystal structure details (crystal graph)
	- `embedding_path`: path where the element embedding JSON files are saved
	- `embedding_type`: element embedding type. Available embedding types: mat2vec, onehot, cgcnn, megnet16, mlelmo
	"""
	def __init__(self,
				train_path,
				val_path,
				test_path,
				model_path='saved_models/best_model_gnn',
				learning_rate= 3e-4,
				epochs=1200,
				batch_size=128,
				patience=300,
				channels=200,
				aggregate_type='mean',
				mae_loss=False,
				train=False,
				test=False,
				pred_func=False,
				is_pickle=False,
				threshold_radius=4,
				use_crystal_structure=False,
				embedding_path='data/embeddings/',
				embedding_type='mat2vec',
				max_no_atoms=500):

		self.train_path = train_path
		self.val_path = val_path
		self.test_path = test_path
		self.model_path = model_path
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.batch_size = batch_size
		self.patience = patience
		self.channels = channels
		self.aggregate_type = aggregate_type
		self.mae_loss = mae_loss
		self.train = train
		self.test = test
		self.pred_func = pred_func
		self.is_pickle = is_pickle
		self.threshold_radius = threshold_radius
		self.use_crystal_structure = use_crystal_structure
		self.embedding_path = embedding_path
		self.embedding_type = embedding_type
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
								threshold_radius=self.threshold_radius,
								use_crystal_structure=self.use_crystal_structure,
								max_no_atoms=self.max_no_atoms)

		## get scaler attribute after fitting on training data
		self.scaler = dataset_tr.scaler
		scaler_dict = self.scaler.state_dict()
		os.makedirs('saved_models/best_model_gnn', exist_ok=True)
		json.dump(scaler_dict, open("saved_models/best_model_gnn/scaler_dict.json", 'w' ))	# save the state of scaler
		
		dataset_val = DataLoader(data_path=self.val_path, 
								is_train=False,
								pred_func=self.pred_func,
								is_pickle=self.is_pickle,
								scaler=self.scaler,
								embedding_path=self.embedding_path,
								embedding_type=self.embedding_type,
								use_crystal_structure=self.use_crystal_structure,
								max_no_atoms=self.max_no_atoms)

		loader_tr = DisjointLoader(dataset_tr, batch_size=self.batch_size, epochs=self.epochs)
		loader_val = DisjointLoader(dataset_val, batch_size=self.batch_size, epochs=1)

		n_out = dataset_tr.n_labels  # Dimension of the target
		if self.mae_loss:
			robust = False
		else:
			robust = True			

		model = Finder(channels=self.channels,
				n_out=n_out,
				robust=robust,
				aggregate_type=self.aggregate_type,
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
				# 									Fit model
				################################################################################
				@tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
				def train_step(inputs, target):
					with tf.GradientTape() as tape:
						predictions = model(inputs, training=True)
						target=tf.cast(target, tf.float32)
						loss = loss_fn(target, predictions) + sum(model.losses)
						if self.mae_loss:
							mae = loss_fn(target, predictions)
						else:
							mae = robust_mae.mean_absolute_error(target, predictions)
					gradients = tape.gradient(loss, model.trainable_variables)
					optimizer.apply_gradients(zip(gradients, model.trainable_variables))

					return loss, mae

				train_mae = []
				validation_mae = []
				step = loss = 0
				tr_mae = 0
				validation_data = list(loader_val)
				epoch_no = 1
				best_val_mae = 1e6
				for batch in loader_tr:
					step += 1
					l, tmae = train_step(*batch)
					loss+=l
					tr_mae+=tmae
					if step == loader_tr.steps_per_epoch:
						val_loss = 0
						val_mae = 0
						for batch_val in validation_data:
							val_inputs, val_targets = batch_val
							val_predictions = model(val_inputs, training=False)
							val_loss += loss_fn(val_targets, val_predictions)
							if self.mae_loss:
								val_mae+=loss_fn(val_targets, val_predictions)
							else:
								val_mae += robust_mae.mean_absolute_error(val_targets, val_predictions)
						
						step = 0
						K.set_value(optimizer.learning_rate, optimizer.lr*0.999) # reduce learning rate
						print('\nEpoch: ', epoch_no)
						print("Training Loss: {:.5f} \t Validation Loss: {:.5f}\n".format(loss / loader_tr.steps_per_epoch, val_loss / loader_val.steps_per_epoch))
						print("Training MAE: {:.5f} \t Validation MAE: {:.5f}\n".format(tr_mae / loader_tr.steps_per_epoch, val_mae / loader_val.steps_per_epoch))

						train_mae.append(tr_mae/loader_tr.steps_per_epoch)
						validation_mae.append(val_mae/loader_val.steps_per_epoch)

						if val_mae/loader_val.steps_per_epoch < best_val_mae:
							# save current best model and scaler metadata
							model.save('saved_models/best_model_gnn',save_format='tf')							

						if len(validation_mae) > self.patience:
							if validation_mae[-(self.patience+1)] < min(validation_mae[-self.patience:]):
								print(f'\nEarly stopping. No validation loss '
		                      			f'improvement in {self.patience} epochs.')
								break

						with open('results/history.csv', 'a+') as file:
							file.write(str(epoch_no)+','+str((tr_mae/loader_tr.steps_per_epoch).numpy())+','+str((val_mae/loader_val.steps_per_epoch).numpy())+'\n')

						epoch_no+=1
						loss = 0
						tr_mae = 0
				tm = [t.numpy() for t in train_mae]
				vm = [v.numpy() for v in validation_mae]
				df = pd.DataFrame({'Train MAE': tm, 'Validation MAE': vm})
				df.to_csv('results/training_history.csv')

				## plotting
				plt.plot(range(1, len(train_mae)+1), train_mae, lw=2, ls='-', c='blue', label='Train')
				plt.plot(range(1, len(validation_mae)+1), validation_mae, lw=2, ls='-', c='red', label='Validation')
				plt.xlabel('Epoch Number', fontsize=14)
				plt.ylabel('Mean Absolute Error', fontsize=14)
				plt.legend()
				plt.tight_layout()
				plt.savefig('results/training_log.png', dpi=100)
				# plt.show()

		except KeyboardInterrupt:
			pass

		if self.test:
			self.test_model()


	def test_model(self):
		################################################################################
		# 							Evaluate model
		################################################################################

		print("\n\nLoading current best model ...\n")
		try:
			model_path = self.model_path
			model = tf.keras.models.load_model(model_path)
		except:
			print('No model exists. Please run with --train to train the model first')

		if self.mae_loss:
			loss_fn = MeanAbsoluteError()
		else:
			loss_fn = RobustLoss()

		scaler_dict = json.load(open("{0}/scaler_dict.json".format(self.model_path)))
		self.scaler.load_state_dict(state_dict=scaler_dict) # update scaler
		robust_mae = RobustMAE(scaler=self.scaler, pred_func=self.pred_func)  

		dataset_te = DataLoader(data_path=self.test_path, 
							is_train=False,
							pred_func=self.pred_func,
							is_pickle=self.is_pickle,
							scaler=self.scaler,
							embedding_path=self.embedding_path,
							embedding_type=self.embedding_type,
							use_crystal_structure=self.use_crystal_structure,
							max_no_atoms=self.max_no_atoms)

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
			loss += loss_fn(target, predictions)
			if self.mae_loss:
				predictions_list.extend(list(predictions.numpy()))
			else:
				predictions_list.extend(list(tf.split(predictions, 2, axis=-1)[0].numpy()))
				uncertainity_list.extend(list(tf.split(predictions, 2, axis=-1)[1].numpy()))
				test_mae += robust_mae.mean_absolute_error(target, predictions)
		loss /= loader_te.steps_per_epoch
		print("Test Loss: {:.5f}".format(loss / loader_te.steps_per_epoch))
		# print("Test MAE denormed: {:.5f}".format(test_mae / loader_te.steps_per_epoch))

		preds = np.array(predictions_list).flatten()
		sigmas = np.array(uncertainity_list).flatten()
		predictions = self.scaler.denorm(preds)
		uncertainities = tf.math.exp(sigmas) * self.scaler.std
		
		df_in = self.get_valid_compounds(pd.read_csv(self.test_path))
		targets = np.array(df_in['target'])
		
		mae = mean_absolute_error(predictions, targets)
		r2 = r2_score(predictions, targets)
		mse = mean_squared_error(predictions, targets)
		
		print("Test MAE: {:.5f}".format(mae))
		print("Test R2 Score: {:.5f}".format(r2))
		print("Test RMSE: {:.5f}".format(np.sqrt(mse)))
		
		df_preds = pd.DataFrame({'formula': df_in['formula'].tolist(), 'target': targets, 'prediction': predictions, 'uncertainity': uncertainities})
		df_preds.to_csv("results/test_results.csv", index=False)

	def get_valid_compounds(self, df):
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

		return df



def argument_parser():
	'''
	parse input arguments
	'''
	parser = argparse.ArgumentParser(
	 description=("Formula graph self-attention network for materials discovery")
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
		default=1200,
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
		'--channels',
		type=int,
		default=200,
		help='Dimension of the internal node representation of the graph layer'
		)
	parser.add_argument(
		'--aggregate-type',
		type=str,
		default='mean',
		help='A permutation invariant function to aggregate messages (e.g. mean, sum, min, max)'
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
		'--threshold-radius', 
		type=int,
		default=4,
		help='''Atoms located at a distance less than the threshold radius are bonded in the crystal graph''' 
		)
	parser.add_argument(
		'--use-crystal-structure', 
		help='''Use the crystal structures to learn material properties. 
			This requires a column 'cif' in train.csv, val.csv and test.csv files containing the cif of all crystals in string format. 
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
		help='Element embedding type. Available embedding types: mat2vec, onehot, cgcnn, megnet16, mlelmo', 
		)
	parser.add_argument(
		'--max-no-atoms',
		type=int,
		default=500,
		help='Maximum number of atoms that can be in the integer formula graph (E.g. BaTiO3 has 5 atoms)'
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
