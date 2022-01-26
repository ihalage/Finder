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
  model.compile(optimizer='adam', loss=tf.keras.losses.RobustLoss())
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
  sigma = tf.clip_by_value(sigma,-10.0,10.0) # evade exploding gradient
  loss =  np.sqrt(2.0) * K.abs(mean - y_true) * K.exp(-sigma)  + sigma
  return K.mean(loss)


class RobustMAE():
	def __init__(self,
				scaler,
				pred_func=False):
		self.scaler = scaler
		self.pred_func = pred_func

	def mean_absolute_error(self, y_true, y_pred):
		mean, sigma = tf.split(y_pred, 2, axis=-1)
		if self.pred_func:
			mae = K.abs(mean - y_true)
		else:
			mae = K.abs(self.scaler.denorm(mean) - self.scaler.denorm(y_true))
		return K.mean(mae)