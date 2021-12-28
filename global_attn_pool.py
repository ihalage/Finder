import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import constraints, initializers, regularizers
from tensorflow.keras.layers import Dense, Layer
from spektral.layers.convolutional.message_passing import MessagePassing
from tensorflow.keras import layers
from spektral.layers import ops
from tensorflow.keras.constraints import max_norm, min_max_norm


class GlobalPool(Layer):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.supports_masking = True
        self.pooling_op = None
        self.batch_pooling_op = None

    def build(self, input_shape):
        if isinstance(input_shape, list) and len(input_shape) == 2:
            self.data_mode = "disjoint"
        else:
            if len(input_shape) == 2:
                self.data_mode = "single"
            else:
                self.data_mode = "batch"
        super().build(input_shape)

    def call(self, inputs):
        if self.data_mode == "disjoint":
            X = inputs[0]
            I = inputs[1]
            if K.ndim(I) == 2:
                I = I[:, 0]
        else:
            X = inputs

        if self.data_mode == "disjoint":
            return self.pooling_op(X, I)
        else:
            return self.batch_pooling_op(
                X, axis=-2, keepdims=(self.data_mode == "single")
            )

    def compute_output_shape(self, input_shape):
        if self.data_mode == "single":
            return (1,) + input_shape[-1:]
        elif self.data_mode == "batch":
            return input_shape[:-2] + input_shape[-1:]
        else:
            # Input shape is a list of shapes for X and I
            return input_shape[0]


class GlobalAttnAvgPool(MessagePassing):
    r"""
    A node-attention global pooling layer. Pools a graph by learning attention
    coefficients to sum node features.
    This layer computes:
    $$
        \alpha = \textrm{softmax}( \X \a); \\
        \X' = \sum\limits_{i=1}^{N} \alpha_i \cdot \X_i
    $$
    where \(\a \in \mathbb{R}^F\) is a trainable vector. Note that the softmax
    is applied across nodes, and not across features.
    **Mode**: single, disjoint, mixed, batch.
    **Input**
    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - Graph IDs of shape `(n_nodes, )` (only in disjoint mode);
    **Output**
    - Pooled node features of shape `(batch, n_node_features)` (if single mode, shape will
    be `(1, n_node_features)`).
    **Arguments**
    - `attn_kernel_initializer`: initializer for the attention weights;
    - `attn_kernel_regularizer`: regularization applied to the attention kernel
    matrix;
    - `attn_kernel_constraint`: constraint applied to the attention kernel
    matrix;
    """

    def __init__(
        self,
        channels=128,
        attn_kernel_initializer="glorot_uniform",
        attn_kernel_regularizer=None,
        attn_kernel_constraint=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.gate_nn = self.GATENN()
        self.message_nn = self.MESSAGENN()

    def build(self, input_shape):
        assert len(input_shape) >= 2
        if isinstance(input_shape, list) and len(input_shape) == 2:
            self.data_mode = "disjoint"
            F = input_shape[0][-1]
        else:
            if len(input_shape) == 2:
                self.data_mode = "single"
            else:
                self.data_mode = "batch"
            F = input_shape[-1]
        # Attention kernels
        # self.attn_kernel = self.add_weight(
        #     shape=(F, 1),
        #     initializer=self.attn_kernel_initializer,
        #     regularizer=self.attn_kernel_regularizer,
        #     constraint=self.attn_kernel_constraint,
        #     name="attn_kernel",
        # )
        self.built = True

    def unsorted_segment_softmax(self, x, indices, n_nodes=None):
        """
        Applies softmax along the segments of a Tensor. This operator is similar
        to the tf.math.segment_* operators, which apply a certain reduction to the
        segments. In this case, the output tensor is not reduced and maintains the
        same shape as the input.
        :param x: a Tensor. The softmax is applied along the first dimension.
        :param indices: a Tensor, indices to the segments.
        :param n_nodes: the number of unique segments in the indices. If `None`,
        n_nodes is calculated as the maximum entry in the indices plus 1.
        :return: a Tensor with the same shape as the input.
        """
        n_nodes = tf.reduce_max(indices) + 1 if n_nodes is None else n_nodes
        e_x = tf.exp(
            x - tf.gather(tf.math.unsorted_segment_max(x, indices, n_nodes), indices)
        )
        # e_x = tf.exp(x)
        

        # l = K.print_tensor(e_x, message='e_x = ')
        # tf.debugging.check_numerics(e_x, "Global pooling e_x have nan!")
        # global_attr = tf.math.unsorted_segment_sum(e_x, indices, n_nodes)
        # _,_,nodes_list = tf.unique_with_counts(self.index_i)
        # gat = tf.repeat(global_attr, nodes_list, axis=-2)
        # seg_sum = tf.math.unsorted_segment_sum(e_x, indices, n_nodes)
        
        # tf.debugging.check_numerics(seg_sum, "Global pooling seg_sum have nan!")
        # gather = tf.gather(
        #     # tf.math.unsorted_segment_sum(e_x, indices, n_nodes) + 1e-9, indices
        #     tf.math.segment_sum(e_x, indices) + 1e-8, indices
        # )
        # tf.debugging.check_numerics(gather, "Global pooling gather have nan!")
        # l = K.print_tensor(gather, message='gather = ')
        


        e_x /= tf.gather(
            tf.math.unsorted_segment_sum(e_x, indices, n_nodes) + 1e-9, indices
            # tf.math.segment_sum(e_x, indices) + 1e-8, indices
        )
        return e_x

    def GATENN(self):
        model = tf.keras.Sequential([
            # layers.BatchNormalization(),
        # layers.Conv1D(2, 3, activation="relu", name="layer1"),
        # layers.Conv1D(128, 3, activation="relu", name="layer2"),
        # layers.Conv1D(256, 3, activation="relu", name="layer3"),
        # layers.Flatten(),#, kernel_constraint=min_max_norm(min_value=1e-40, max_value=1.0)
        layers.Dense(256, activation="relu", name="layer4", kernel_regularizer=regularizers.l2(1e-6), kernel_constraint=min_max_norm(min_value=1e-30, max_value=1.0)),
        # layers.Dense(64, activation="relu", name="layer2", kernel_regularizer=regularizers.l2(1e-6)),
        # layers.Dense(128, activation="relu", name="layer3"),
        # layers.Dense(256, activation="relu", name="layer4"),
        layers.Dense(self.channels, name="layer5", kernel_constraint=min_max_norm(min_value=1e-30, max_value=1.0))])
        return model

    def MESSAGENN(self):
        model = tf.keras.Sequential([
            # layers.BatchNormalization(),
        # layers.Conv1D(2, 3, activation="relu", name="layer1"),
        # layers.Conv1D(128, 3, activation="relu", name="layer2"),
        # layers.Conv1D(256, 3, activation="relu", name="layer3"),
        # layers.Flatten(),
        layers.Dense(256, activation="relu", name="layer4", kernel_regularizer=regularizers.l2(1e-6)),
        # layers.Dense(64, activation="relu", name="layer2", kernel_regularizer=regularizers.l2(1e-6)),
        # layers.Dense(128, activation="relu", name="layer3"),
        # layers.Dense(256, activation="relu", name="layer4"),
        layers.Dense(self.channels, name="layer5")])
        return model


    def call(self, inputs):
        if self.data_mode == "disjoint":
            X, I = inputs
            if K.ndim(I) == 2:
                I = I[:, 0]
        else:
            X = inputs

        # # attn_coeff = K.dot(X, self.attn_kernel)
        # # attn_coeff = K.squeeze(attn_coeff, -1)
        # # attn_coeff = K.softmax(attn_coeff)
        # if self.data_mode == "single":
        #     output = K.dot(attn_coeff[None, ...], X)
        # elif self.data_mode == "batch":
        #     output = K.batch_dot(attn_coeff, X)
        # else:
        #     # output = attn_coeff[:, None] * X
        #     output = tf.math.segment_mean(output, I)

        # x_i = self.get_i(X)
        # x_j = self.get_j(X)
        # print(X.shape)
        gate = self.gate_nn(X)
        # tf.debugging.check_numerics(gate, "gate weights have nan!")
        # gate=X
        self.n_nodes = tf.shape(X)[-2]
        # neighbors_mean = (x_i+x_j)/2.0
        aij = self.unsorted_segment_softmax(gate, I, self.n_nodes)
        # tf.debugging.check_numerics(aij, "Global pool aij weights have nan!")
        # aij = scatter_mean(aij, self.index_i, self.n_nodes)
        # print('X: ', X.shape)
        # print('aij: ', aij.shape)
        x = self.message_nn(X)
        # x=X
        output = tf.math.segment_sum(x*aij, I)
        # output = tf.math.segment_max(x*aij, I)

        # one_string = tf.strings.format("{}\n", output, summarize=-1)
        # tf.io.write_file('global_attribute', one_string, name=None)



        return output

    def get_config(self):
        config = {
            "attn_kernel_initializer": self.attn_kernel_initializer,
            "attn_kernel_regularizer": self.attn_kernel_regularizer,
            "attn_kernel_constraint": self.attn_kernel_constraint,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))



class GlobalAttnSumPool(GlobalPool):
    r"""
    A node-attention global pooling layer. Pools a graph by learning attention
    coefficients to sum node features.
    This layer computes:
    $$
        \alpha = \textrm{softmax}( \X \a); \\
        \X' = \sum\limits_{i=1}^{N} \alpha_i \cdot \X_i
    $$
    where \(\a \in \mathbb{R}^F\) is a trainable vector. Note that the softmax
    is applied across nodes, and not across features.
    **Mode**: single, disjoint, mixed, batch.
    **Input**
    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - Graph IDs of shape `(n_nodes, )` (only in disjoint mode);
    **Output**
    - Pooled node features of shape `(batch, n_node_features)` (if single mode, shape will
    be `(1, n_node_features)`).
    **Arguments**
    - `attn_kernel_initializer`: initializer for the attention weights;
    - `attn_kernel_regularizer`: regularization applied to the attention kernel
    matrix;
    - `attn_kernel_constraint`: constraint applied to the attention kernel
    matrix;
    """

    def __init__(
        self,
        attn_kernel_initializer="glorot_uniform",
        attn_kernel_regularizer=None,
        attn_kernel_constraint=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        if isinstance(input_shape, list) and len(input_shape) == 2:
            self.data_mode = "disjoint"
            F = input_shape[0][-1]
        else:
            if len(input_shape) == 2:
                self.data_mode = "single"
            else:
                self.data_mode = "batch"
            F = input_shape[-1]
        # Attention kernels
        self.attn_kernel = self.add_weight(
            shape=(F, 1),
            initializer=self.attn_kernel_initializer,
            regularizer=self.attn_kernel_regularizer,
            constraint=self.attn_kernel_constraint,
            name="attn_kernel",
        )
        self.built = True

    def call(self, inputs):
        if self.data_mode == "disjoint":
            X, I = inputs
            if K.ndim(I) == 2:
                I = I[:, 0]
        else:
            X = inputs
        attn_coeff = K.dot(X, self.attn_kernel)
        attn_coeff = K.squeeze(attn_coeff, -1)
        attn_coeff = K.softmax(attn_coeff)
        if self.data_mode == "single":
            output = K.dot(attn_coeff[None, ...], X)
        elif self.data_mode == "batch":
            output = K.batch_dot(attn_coeff, X)
        else:
            output = attn_coeff[:, None] * X
            output = tf.math.segment_sum(output, I)

        return output

    def get_config(self):
        config = {
            "attn_kernel_initializer": self.attn_kernel_initializer,
            "attn_kernel_regularizer": self.attn_kernel_regularizer,
            "attn_kernel_constraint": self.attn_kernel_constraint,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

