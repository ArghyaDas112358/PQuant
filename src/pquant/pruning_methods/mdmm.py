import keras
from keras import ops
import abc

@ops.custom_gradient
def flip_gradient(weight):
    """
    An identity function that flips the sign of the gradient during
    the backward pass. This is used to perform gradient ascent using a
    standard gradient descent optimizer.
    """
    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        return -upstream
    return weight, grad


@keras.utils.register_keras_serializable(name = "Constraint")
class Contraint(keras.layers.Layer):
    """
    Abstract base class for a constraint.
    Each constraint has its own Lagrange multiplier (lmbda) and calculates
    its own penalty, which it adds to the total model loss.
    """
    def __init__(self, scale=1.0, damping=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = self.add_weight(
            name='scale',
            shape=(),
            initializer=lambda shape, dtype: ops.convert_to_tensor(scale, dtype=dtype),
            trainable=False
        )
        self.damping = self.add_weight(
            name='damping',
            shape=(),
            initializer=lambda shape, dtype: ops.convert_to_tensor(damping, dtype=dtype),
            trainable=False
        )
        self.lmbda = self.add_weight(
            name=self.name + '_lmbda',
            shape=(),
            initializer=keras.initializers.Zeros(),
            trainable=True
        )
    
    def call(self, inputs):
        fn_value = self.ctr_fn(inputs)
        infeasibility = self.ctr_infeasibility(fn_value)
        l_term = ops.maximum(self.lmbda, 0.0) * infeasibility
        damp_term = self.damping * ops.square(infeasibility) / 2
        additional_loss = self.scale * (l_term + damp_term)
        return additional_loss
    
    @abc.abstractmethod
    def ctr_fn(self, inputs):
        """This function should be implemented by subclasses to compute the constraint function."""
        raise NotImplementedError("Subclasses should implement ctr_fn() method")
    
    @abc.abstractmethod
    def ctr_infeasibility(self, ctr_fn):
        """This function should be implemented by subclasses to compute the constraint infeasibility."""
        raise NotImplementedError("Subclasses should implement ctr_infeasibility() method")
    
@keras.utils.register_keras_serializable(name = "EqL1Constraint")
class EqL1Constraint(Contraint):
    def __init__(self, layer, target_val, epsilon=1e-3, scale=1.0, damping=1.0, **kwargs):    
        super().__init__(scale=scale, damping=damping, **kwargs)
        self.layer = layer
        self.target_val = target_val
        self.epsilon = epsilon
        


class MDMM(keras.layers.Layer):
    def __init__(self, config, layer_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        
        self.layer_type = layer_type
        
    def build(self, input_shape):
        
        self.target_sparsity = self.config["pruning_parameters"]["target_sparsity"]
        self.epsilon = self.config["pruning_parameters"]["epsilon"]
        self.lmbda = self.add_weight(
            name = 'lmbda',
            shape = (),
            initializer = "zeros",
            trainable = True
        )
        self.mask = ops.ones(input_shape)
        
    def call(self, weight):
        return self.get_mask(weight) * weight
    
    def get_mask(self, weight):
        return self.mask
    
    def get_hard_mask(self, weight):
        return self.mask
    
    def calculate_additional_loss(self):
        
        
'''
import tensorflow as tf
from tensorflow.keras import layers
import abc

@tf.keras.utils.register_keras_serializable(name='Constraint')
class Constraint(layers.Layer):
    """Base class for constraints."""
    def __init__(self, scale=1.0, damping=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = self.add_weight(
            name='scale',
            shape=(),
            initializer=tf.constant_initializer(scale),
            trainable=False
        )
        self.damping = self.add_weight(
            name='damping',
            shape=(),
            initializer=tf.constant_initializer(damping),
            trainable=False
        )
        self.lmbda = self.add_weight(
            name=self.name + '_lmbda',
            shape=(),
            initializer=tf.zeros_initializer(),
            trainable=True
        )

    def call(self, inputs):
        fn_value = self.fn(inputs)
        inf = self.infeasibility(fn_value)
        l_term = tf.math.maximum(self.lmbda, 0.0) * inf
        damp_term = self.damping * tf.square(inf) / 2
        penalty = self.scale * (l_term + damp_term)
        return penalty

    @abc.abstractmethod
    def fn(self, inputs):
        raise NotImplementedError("Subclasses should implement fn() method")

    @abc.abstractmethod
    def infeasibility(self, fn_value):
        raise NotImplementedError("Subclasses should implement infeasibility() method")

    @abc.abstractmethod
    def compute_update_lmbda(self):
        raise NotImplementedError("Subclasses should implement compute_update_lmbda() method")

    def get_config(self):
        config = super().get_config()
        config.update({
            "scale": self.scale.numpy(),
            "damping": self.damping.numpy(),
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(name='EqL1Constraint')
class EqL1Constraint(Constraint):
    def __init__(self, layer, target_sparsity, scale=1.0, damping=1.0, epsilon=1e-5, lr_multiplier=1.0, **kwargs):
        super().__init__(scale, damping, **kwargs)

        assert 0 <= target_sparsity <= 1, "target_sparsity must be between 0 and 1"
        self.target_sparsity = target_sparsity
        self.epsilon = epsilon
        self.lr_multiplier = lr_multiplier

        self.weights_list = []
        if isinstance(layer, list):
            for l in layer:
                self.weights_list.append(l.weights[0])
        else:
            self.weights_list.append(layer.weights[0])

    def fn(self, inputs):
        weights_concat = tf.concat([tf.reshape(w, [-1]) for w in self.weights_list], axis=0)
        num_weights = tf.cast(tf.size(weights_concat), tf.float32)
        zero_weights = tf.less_equal(tf.abs(weights_concat), self.epsilon)
        zero_count = tf.reduce_sum(tf.cast(zero_weights, tf.float32))
        l1_term = tf.reduce_mean(tf.abs(weights_concat))

        target_zero_count = tf.math.ceil(num_weights * self.target_sparsity)
        factor = (target_zero_count - zero_count) / num_weights

        fn_value = tf.math.maximum(factor, 0.0) * l1_term
        return fn_value

    def infeasibility(self, fn_value):
        return abs(0.0 - fn_value)

    def compute_update_lmbda(self):
        weights_concat = tf.concat([tf.reshape(w, [-1]) for w in self.weights_list], axis=0)
        num_weights = tf.cast(tf.size(weights_concat), tf.float32)
        zero_weights = tf.less_equal(tf.abs(weights_concat), self.epsilon)
        zero_count = tf.reduce_sum(tf.cast(zero_weights, tf.float32))

        target_zero_count = tf.math.ceil(num_weights * self.target_sparsity)
        factor = (target_zero_count - zero_count) / num_weights
        factor_2 = tf.math.pow(factor, 2)

        new_update_lmbda = tf.where(
            (factor >= 1e-6) & (factor_2 > 1e-6),
            self.lr_multiplier * factor_2,
            1e-6
        )
        return new_update_lmbda

    def get_config(self):
        config = super().get_config()
        # Layer is not serializable, so we store its name
        config.update({
            "layer_name": self.layer.name,
            "target_sparsity": self.target_sparsity,
            "epsilon": self.epsilon,
            "lr_multiplier": self.lr_multiplier
        })
        return config
'''