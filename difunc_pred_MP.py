import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import rcParams

from pymatgen import Composition, Element, MPRester
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from pymatgen import Composition

from difunc_NN import DiFuncNN


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



class InferDifunc(object):
	def __init__(self):
	# 	# self.df_X = pd.read_pickle('difunc_data/df_X.pkl')
		self.dfnn = DiFuncNN()
	# 	self.df_X = self.dfnn.df_X
		self.columns = self.dfnn.columns

	def vectorize_df(self, df):
		dfv = self.dfnn.vectorize(df)
		dfv.to_pickle('difunc_data/MP_data_no_jarvis_pred.pkl')
	def get_MP_data(self):
		df_MP = pd.read_pickle('MP_data/MP_data.pkl')
		# print(df_MP.info())
		df = self.dfnn.featurize(df_MP)
		df = df[['full_formula']+list(self.columns)]
		df.drop_duplicates(keep='first', inplace=True)
		df_X = pd.read_pickle('difunc_data/df_X.pkl')
		df_X = df_X[['full_formula']+list(self.columns)]
		# print(df)
		# df_pred = pd.concat([df,df_X], ignore_index=True)#.drop_duplicates(keep=False)
		# print(df_pred.info())
		# print(df_pred)
		df_all = df.merge(df_X.drop_duplicates(), how='left',indicator=True)
		df_final = df_all[df_all['_merge'] == 'left_only'].drop(['_merge'], axis=1)
		print(df_final.info())
		pd.to_pickle(df_final, 'difunc_data/df_pred.pkl')

	def predict_MP(self, df, n_samples=20):
		# df = pd.read_pickle('difunc_data/df_pred.pkl')
		# X = np.expand_dims(np.array(df.drop(['full_formula'], axis=1)), axis=-1)
		df_icsd = pd.read_pickle('MP_data/MP_ICSD_no_jarvis.pkl')
		X = np.expand_dims(np.array(df[range(103)]), axis=-1)
		print(df.shape)
		model_re = tf.keras.models.load_model('saved_models/Rex_best_model_skipcnn_clipped.h5', compile=False)
		model_im = tf.keras.models.load_model('saved_models/Imx_best_model_skipcnn_clipped.h5', compile=False)
		y_pred_real = model_re.predict(X)
		y_pred_imag = model_im.predict(X)
		plasma_freq_list = []
		imag_plasma_list = []
		for i in range(df.shape[0]):
			# print(i)
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

		df['plasma_frequency'] = plasma_freq_list
		df['epsilon_imaginary_plasma'] = imag_plasma_list
		df['in_ICSD'] = df.apply(lambda x: x.red_formula in df_icsd['red_formula'].tolist(), axis=1)
		df_sorted = df.sort_values(by=['plasma_frequency'])
		df_sorted[['full_formula', 'plasma_frequency', 'epsilon_imaginary_plasma', 'e_above_hull', 'formation_energy_per_atom', 'in_ICSD']].to_csv('MP_data/MP_predictions_no_jarvis.csv', sep='\t', index=False)


		# for i in range(n_samples):
		# 	plt.margins(x=0)
		# 	plt.tight_layout()
		# 	# plt.plot(np.array(df_real.iloc[:, [0]]), np.array(df_real.iloc[:, [1]]), color='blue', linewidth=2, label='$\epsilon_r$')
		# 	# plt.plot(np.array(df_imag.iloc[:, [0]]), np.array(df_imag.iloc[:, [1]]), color='red', linewidth=2, label='$\epsilon_i$')

		# 	plt.plot(np.linspace(0, 30, 3000), y_pred_real[i+80], lw=2, ls='--', c='blue', label='$\epsilon_r$ prediction')
		# 	plt.plot(np.linspace(0, 30, 3000), y_pred_imag[i+80], lw=2, ls='--', c='red', label='$\epsilon_i$ prediction')

		# 	plt.ylabel('Dielectric constant', fontsize=16)
		# 	plt.xlabel('Frequency (eV)', fontsize=16)
		# 	plt.legend(fontsize=14)
		# 	plt.xlim(0, 15)
		# 	composition = df.iloc[i+80]['full_formula']
		# 	plt.title(composition + " - Dielectric constant vs frequency", fontsize=16)
		# 	# plt.savefig(composition + '.png', dpi=800)
		# 	plt.show()


	def get_low_loss_ENZ(self):
		df = pd.read_csv('MP_data/MP_predictions_no_jarvis.csv', sep='\t')
		df = df[(df.plasma_frequency >0.4959) & (df.plasma_frequency <3.1) & (df.epsilon_imaginary_plasma<2)]
		# df.to_csv('MP_data/MP_low_loss_ENZ.csv', sep='\t', index=False)
		def get_enz_range(row):
			enz_range = 0
			if row.plasma_frequency <= 0.4959:
				enz_range = 'low frequency'
			elif row.plasma_frequency <= 1.549:
				enz_range = 'near IR'
			elif row.plasma_frequency <= 3.1:
				enz_range = 'visible'
			else:
				enz_range = 'high frequency'
			return enz_range


		df['ENZ_range'] = df.apply(get_enz_range, axis=1)
		df.to_csv('MP_data/MP_low_loss_ENZ.csv', sep='\t', index=False)

	def analyse_predictions(self):
		df = pd.read_csv('MP_data/MP_predictions.csv', sep='\t')
		fig, ax = plt.subplots(1, 1, figsize=(16,8))
		ax.scatter(df['plasma_frequency'], df['epsilon_imaginary_plasma'], marker='x', s=2)
		plt.show()

	def make_predictions(self, df):
		X = np.expand_dims(np.array(df[self.columns]), axis=-1)
		print(df.shape)
		model_re = tf.keras.models.load_model('saved_models/Rex_best_model_cnn_clipped.h5', compile=False)
		model_im = tf.keras.models.load_model('saved_models/Imx_best_model_cnn_clipped.h5', compile=False)
		y_pred_real = model_re.predict(X)
		y_pred_imag = model_im.predict(X)
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

		df['plasma_frequency'] = plasma_freq_list
		df['epsilon_imaginary_plasma'] = imag_plasma_list
		df_sorted = df.sort_values(by=['plasma_frequency'])
		df_sorted[['full_formula', 'Ed', 'Ef', 'plasma_frequency', 'epsilon_imaginary_plasma']].to_csv('pred_data/SMACT_predictions_1M.csv', sep='\t', index=False)




infer = InferDifunc()
# infer.vectorize_df(pd.read_pickle('MP_data/MP_data_no_jarvis.pkl'))
# infer.get_MP_data()
# df = pd.read_pickle('difunc_data/MP_data_no_jarvis_pred.pkl')
# infer.predict_MP(df)
infer.get_low_loss_ENZ()
# infer.analyse_predictions()
# df = pd.read_pickle('pred_data/SMACT_1M_Ef_Ed.pkl')
# infer.make_predictions(df.iloc[:200000])

