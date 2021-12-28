import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.util import dispatch


class ElemProp():

	def __init__(self):

		self.ox_st_ion_radii = {
								'H': {-1: 1.34, 1: 0.25},
								'Li':{1: 0.90},
								'Be':{2: 0.59},
								'B' :{3: 0.41},
								'C' :{4: 0.30}, 
								'N' :{-3: 1.32, 3: 0.30, 5: 0.27},
								'O' :{-2: 1.26},
								'F' :{-1: 1.19},
								'Na':{1: 1.16},
								'Mg':{2: 0.86},
								'Al':{3: 0.675},
								'Si':{4: 0.54},
								'P' :{3: 0.58},
								'S' :{4: 0.51},
								'Cl':{-1: 1.67, 5: 0.26, 7: 0.41},
								'K' :{1: 1.52},
								'Ca':{2: 1.14},
								'Sc':{3: 0.885},
								'Ti':{4: 0.745},
								'V' :{5: 0.68},
								'Cr':{3: 0.755},
								'Mn':{2: 0.81},
								'Fe':{3: 0.69},
								'Co':{2: 0.79},# high spin for oxi state 4
								'Ni':{2: 0.83},
								'Cu':{1: 0.91},
								'Zn':{2: 0.88},
								'Ga':{3: 0.76},
								'Ge':{4: 0.67},
								'As':{3: 0.72},
								'Se':{-2: 1.84, 4: 0.64, 6: 0.56},
								'Br':{-1: 1.82, 3: 0.73, 5: 0.45, 7: 0.53},
								'Rb':{1: 1.66},
								'Sr':{2: 1.32},
								'Y' :{3: 1.04},
								'Zr':{4: 0.86},
								'Nb':{5: 0.78},
								'Mo':{6: 0.73},
								'Tc':{4: 0.785},
								'Ru':{3: 0.82},
								'Rh':{3: 0.805},
								'Pd':{2: 1.00},
								'Ag':{1: 1.29},
								'Cd':{2: 1.09},
								'In':{3: 0.94},
								'Sn':{4: 0.83},
								'Sb':{5: 0.74},
								'Te':{-2: 2.07, 4: 1.11, 6: 0.70},
								'I' :{-1: 2.06, 5: 1.09, 7: 0.67},
								'Xe':{8: 0.62},
								'Cs':{1: 1.81},
								'Ba':{2: 1.49},
								'La':{3: 1.172}, 
								'Ce':{3: 1.15},
								'Pr':{3: 1.15},
								'Nd':{3: 1.123},
								'Pm':{3: 1.11},
								'Sm':{3: 1.098},
								'Eu':{3: 1.087},
								'Gd':{3: 1.078},
								'Tb':{3: 1.063},
								'Dy':{3: 1.052},
								'Ho':{3: 1.041},
								'Er':{3: 1.03},
								'Tm':{3: 1.02},
								'Yb':{3: 1.008},
								'Lu':{3: 1.001},
								'Hf':{4: 0.85},
								'Ta':{5: 0.78},
								'W' :{6: 0.74},
								'Re':{4: 0.77},
								'Os':{4: 0.77},
								'Ir':{4: 0.765},
								'Pt':{2: 0.94},
								'Au':{1: 1.51},
								'Hg':{2: 1.16},
								'Tl':{1: 1.64},
								'Pb':{2: 1.33, 4: 0.915},
								'Bi':{3: 1.17, 5: 0.9},
								'Po':{4: 1.08},
								'At':{7: 0.76},
								'Fr':{1: 1.94},
								'Ra':{2: 1.62},
								'Ac':{3: 1.26},
								'Th':{4: 1.08},
								'Pa':{5: 0.92},
								'U' :{6: 0.87},
								'Np':{5: 0.89},
								'Pu':{4: 1.00},
								'Am':{3: 1.115},
								'Cm':{3: 1.11},
								'Bk':{3: 1.10},
								'Cf':{3: 1.09},
								'Es':{3: 0.928}
								}

		self.covalent_radius = {
								'H' :0.32,
								'He':0.46,
								'B' :0.85,
								'C' :0.75,
								'N' :0.71,
								'O' :0.63,
								'F' :0.64,
								'Ne':0.67,
								'Si':1.16,
								'P' :1.11,
								'S' :1.03,
								'Cl':0.99,
								'Ar':0.96,
								'Ge':1.21,
								'As':1.21,
								'Se':1.16,
								'Br':1.14,
								'Kr':1.17,
								'Sb':1.4,
								'Te':1.36,
								'I' :1.33,
								'Xe':1.31,
								'Po':1.45,
								'At':1.47,
								'Rn':1.42
								}


		self.metallic_radius = {
								'Li':1.52 ,
								'Be':1.12 ,
								'Na':1.86 ,
								'Mg':1.60 ,
								'Al':1.43 ,
								'K' :2.27,
								'Ca':1.97 ,
								'Sc':1.62 ,
								'Ti':1.47 ,
								'V' :1.34 ,
								'Cr':1.28 ,
								'Mn':1.27 ,
								'Fe':1.26 ,
								'Co':1.25 ,
								'Ni':1.24 ,
								'Cu':1.28 ,
								'Zn':1.34 ,
								'Ga':1.35 ,
								'Rb':2.48 ,
								'Sr':2.15 ,
								'Y' :1.80 ,
								'Zr':1.60 ,
								'Nb':1.46 ,
								'Mo':1.39 ,
								'Tc':1.36 ,
								'Ru':1.34 ,
								'Rh':1.34 ,
								'Pd':1.37 ,
								'Ag':1.44 ,
								'Cd':1.51 ,
								'In':1.67 ,
								'Sn':1.58 ,
								'Cs':2.65 ,
								'Ba':2.22 ,
								'La':1.87 ,
								'Ce':1.818 ,
								'Pr':1.824 ,
								'Nd':1.814 ,
								'Pm':1.834 ,
								'Sm':1.804 ,
								'Eu':1.804 ,
								'Gd':1.804 ,
								'Tb':1.773 ,
								'Dy':1.781 ,
								'Ho':1.762 ,
								'Er':1.761 ,
								'Tm':1.759 ,
								'Yb':1.76 ,
								'Lu':1.738 ,
								'Hf':1.59 ,
								'Ta':1.46 ,
								'W' :1.39 ,
								'Re':1.37 ,
								'Os':1.35 ,
								'Ir':1.355 ,
								'Pt':1.385 ,
								'Au':1.44 ,
								'Hg':1.51 ,
								'Tl':1.70 ,
								'Pb':1.75 ,
								'Bi':1.82 ,
								'Fr':2.70 ,
								'Ra':2.15 ,
								'Ac':1.95 ,
								'Th':1.79 ,
								'Pa':1.63 ,
								'U' :1.56 ,
								'Np':1.55 ,
								'Pu':1.59 ,
								'Am':1.73 ,
								'Cm':1.74 ,
								'Bk':1.70 ,
								'Cf':1.86 ,
								'Es':1.86 
								}


	def get_ionic_radius(self, element):
		return self.ox_st_ion_radii[element]

	def get_covalent_radius(self, element):
		return self.covalent_radius[element]

	def get_metallic_radius(self, element):
		return self.metallic_radius[element]


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, log=False):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = 0
        self.std = 0

    def fit(self, tensor, dim=0, keepdim=False):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = np.mean(tensor, dim, keepdims=keepdim)
        self.std = np.std(tensor, dim, keepdims=keepdim)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]


class NormalizeTensor(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, log=False):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = K.variable(0)
        self.std = K.variable(1)

    def fit(self, tensor, dim=0, keepdim=False):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = K.mean(tensor, dim, keepdim)
        self.std = K.std(tensor, dim, keepdim)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return K.dot(normed_tensor, self.std) + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = tf.convert_to_tensor(tf.cast(state_dict["mean"], dtype=tf.float32))
        self.std = tf.convert_to_tensor(tf.cast(state_dict["std"], dtype=tf.float32))


class RobustLoss(LossFunctionWrapper):
  """Computes the Robust Loss between labels and predictions.

  
  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.RobustLoss())
  ```
  """

  def __init__(self,
               reduction=losses_utils.ReductionV2.AUTO,
               name='robust_loss'):
    """Initializes `MeanAbsoluteError` instance.

    Args:
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training)
        for more details.
      name: Optional name for the op. Defaults to 'robust_loss'.
    """
    super(RobustLoss, self).__init__(
        robust_loss, name=name, reduction=reduction)

@keras_export('keras.metrics.robust_loss',
              'keras.metrics.robust',
              'keras.metrics.ROBUST',
              'keras.losses.robust_loss',
              'keras.losses.robust',
              'keras.losses.ROBUST')
@dispatch.add_dispatch_support
def robust_loss(y_true, y_pred):
  """Computes the robust loss between labels and predictions.

  """
  mean, sigma = tf.split(y_pred, 2, axis=-1)
  # log_sigma = tf.math.log(sigma)
  # print('mean', mean)
  # print('sigma',sigma)
  # sigma = tf.math.maximum(sigma, -4)
  sigma = tf.clip_by_value(sigma,-10.0,10.0)
  loss =  np.sqrt(2.0) * K.abs(mean - y_true) * K.exp(-sigma)  + sigma
  # tf.debugging.check_numerics(loss, "Loss/exp(sigma) has nan!")
  # return K.mean(K.abs(mean - y_true))
  return K.mean(loss)


class RobustMAE():
	def __init__(self,
				scaler,
				pred_func=False):
		self.scaler = scaler
		self.pred_func = pred_func
	# def __init__(self,
	# 		   reduction=losses_utils.ReductionV2.AUTO,
	# 		   name='mean_absolute_error'):
	# 	super(RobustMAE, self).__init__(
	# 	self.mean_absolute_error, name=name, reduction=reduction)

	def mean_absolute_error(self, y_true, y_pred):
		mean, sigma = tf.split(y_pred, 2, axis=-1)
		# mae = K.abs(mean - y_true)
		if self.pred_func:
			mae = K.abs(mean - y_true)
		else:
			mae = K.abs(self.scaler.denorm(mean) - self.scaler.denorm(y_true))
		# print(K.mean(mae), self.tensor_scaler.denorm(K.mean(mae)))
		return K.mean(mae)