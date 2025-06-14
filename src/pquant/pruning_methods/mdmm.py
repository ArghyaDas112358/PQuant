# -*- coding: utf-8 -*-
# @Author: Arghya Ranjan Das
# file: src/pquant/pruning_methods/mdmm.py
# modified by:


import keras
from keras import ops
import abc

@ops.custom_gradient
def flip_gradient(weight):
    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        return -upstream
    return weight, grad

# Abstract base class for constraints
@keras.utils.register_keras_serializable(name = "Constraint")
class Constraint(keras.layers.Layer):
    def __init__(self, scale=1.0, damping=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = self.add_weight(
            name='scale',
            shape=(),
            initializer=lambda s, d: ops.convert_to_tensor(scale, dtype=d),
            trainable=False
        )
        self.damping = self.add_weight(
            name='damping',
            shape=(),
            initializer=lambda s, d: ops.convert_to_tensor(damping, dtype=d),
            trainable=False
        )
        self.lmbda = self.add_weight(
            name=f'{self.name}_lmbda',
            shape=(),
            initializer=keras.initializers.Zeros(),
            trainable=True
        )
    
    def calculate_penalty(self, weight):
        """Calculates the penalty from a given infeasibility measure."""
        raw_infeasibility = self.get_infeasibility(weight)
        infeasibility = self.pipe_infeasibility(raw_infeasibility)
        
        ascent_lmbda = flip_gradient(self.lmbda)
        
        l_term = ascent_lmbda * infeasibility
        damp_term = self.damping * ops.square(infeasibility) / 2
        penalty = self.scale * (l_term + damp_term)
        
        return penalty

    @abc.abstractmethod
    def get_infeasibility(self, weight):
        """Must be implemented by subclasses to define the violation."""
        raise NotImplementedError
    
    def pipe_infeasibility(self, infeasibility):
        """Optional transformation of raw infeasibility.
        Default is identity. Subclasses may override."""
        return infeasibility

#-------------------------------------------------------------------
#               Generic Constraint Classes
#-------------------------------------------------------------------

@keras.utils.register_keras_serializable(name = "EqualityConstraint")
class EqualityConstraint(Constraint):
    """Constraint for g(w) == target_value."""
    def __init__(self, metric_fn, target_value = 0.0,**kwargs):
        super().__init__(**kwargs)
        self.metric_fn = metric_fn
        self.target_value = target_value
        
    def get_infeasibility(self, weight):
        metric_value = self.metric_fn(weight)
        infeasibility = metric_value - self.target_value
        return ops.abs(infeasibility)
    
    
@keras.utils.register_keras_serializable(name = "LessThanOrEqualConstraint")
class LessThanOrEqualConstraint(Constraint):
    """Constraint for g(w) <= target_value."""
    def __init__(self, metric_fn, target_value = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.metric_fn = metric_fn
        self.target_value = target_value
        
    def get_infeasibility(self, weight):
        metric_value = self.metric_fn(weight)
        infeasibility = metric_value - self.target_value
        return ops.maximum(infeasibility, 0.0)
    
@keras.utils.register_keras_serializable(name = "GreaterThanOrEqualConstraint")
class GreaterThanOrEqualConstraint(Constraint):
    """Constraint for g(w) >= target_value."""
    def __init__(self, metric_fn, target_value = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.metric_fn = metric_fn
        self.target_value = target_value
        
    def get_infeasibility(self, weight):
        metric_value = self.metric_fn(weight)
        infeasibility = self.target_value - metric_value
        return ops.maximum(infeasibility, 0.0)
    
#-------------------------------------------------------------------
#                   Metric Functions
#-------------------------------------------------------------------

class UnstructuredSparsityMetric:
    """Calculates the ratio of non-zero weights in a tensor."""
    def __init__(self, epsilon=1e-3):
        self.epsilon = epsilon
    def __call__(self, weight):
        num_weights = ops.cast(ops.size(weight), weight.dtype)
        zero_weights = ops.less_equal(ops.abs(weight), self.epsilon)
        zero_count = ops.reduce_sum(ops.cast(zero_weights, weight.dtype))
        sparsity_ratio = zero_count / num_weights
        return sparsity_ratio

class StructuredSparsityMetric:
    """Calculates the ratio of near-zero weight groups (based on Reuse Factor: rf)."""
    def __init__(self, rf=1, epsilon=1e-3):
        self.rf = rf
        self.epsilon = epsilon
    
    def __call__(self, weight):
        original_shape = weight.shape
        w_reshaped = ops.reshape(weight, (original_shape[0], -1))
        num_weights = ops.shape(w_reshaped)[1]
        
        padding = (self.rf - num_weights % self.rf) % self.rf
        w_padded = ops.pad(w_reshaped, [[0, 0], [0, padding]])
        
        groups = ops.reshape(w_padded, (original_shape[0], -1, self.rf))
        group_norms = ops.sqrt(ops.sum(ops.square(groups), axis=-1))
        zero_groups = ops.less_equal(group_norms, self.epsilon)
        num_groups = ops.cast(ops.size(group_norms), "float32")
        
        return ops.reduce_sum(ops.cast(zero_groups, "float32")) / num_groups

#-------------------------------------------------------------------
#                   MDMM Layer
#-------------------------------------------------------------------
    
class MDMM(keras.layers.Layer):
    def __init__(self, config, layer_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config["pruning_parameters"]
        self.layer_type = layer_type
        self.constraint_layer = None
        self.penalty_loss = self.add_weight(shape=(), initializer='zeros', trainable=False)

    
    def build(self, input_shape):
        metric_type = self.config.get("metric_type", "UnstructuredSparsity")
        constraint_type = self.config.get("constraint_type", "GreaterThanOrEqual")
        target_value = self.config.get("target_value", 0.8)
        
        if metric_type == "UnstructuredSparsity":
            metric_fn = UnstructuredSparsityMetric(epsilon=self.config.get("epsilon", 1e-5))
        elif metric_type == "StructuredSparsity":
            metric_fn = StructuredSparsityMetric(rf=self.config["rf"], epsilon=self.config.get("epsilon", 1e-5))
        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")

        common_args = {
            "metric_fn": metric_fn,
            "target_value": target_value,
            "scale": self.config.get("scale", 1.0),
            "damping": self.config.get("damping", 1.0)
        }
        
        if constraint_type == "Equality":
            self.constraint_layer = EqualityConstraint(**common_args)
        elif constraint_type == "LessThanOrEqual":
            self.constraint_layer = LessThanOrEqualConstraint(**common_args)
        elif constraint_type == "GreaterThanOrEqual":
            self.constraint_layer = GreaterThanOrEqualConstraint(**common_args)
        else:
            raise ValueError(f"Unknown constraint_type: {constraint_type}")
        
        self.mask = ops.ones(input_shape)
        self.constraint_layer.build(input_shape)
        super().build(input_shape)
        self.built = True
                    
    def call(self, weight):
        if not self.built:
            self.build(weight.shape)

        penalty_loss_value = self.constraint_layer.calculate_penalty(weight)
        self.penalty_loss.assign(penalty_loss_value)

        return weight 
    
    def get_hard_mask(self, weight):
        epsilon = self.config.get("epsilon", 1e-5)
        return ops.cast(ops.abs(weight) > epsilon, weight.dtype)
    
    def get_layer_sparsity(self, weight):
        return ops.sum(self.get_mask(weight)) / ops.size(weight)

    def calculate_additional_loss(self):
        return self.penalty_loss

    def pre_epoch_function(self, epoch, total_epochs):
        pass

    def pre_finetune_function(self):
        pass

    def post_epoch_function(self, epoch, total_epochs):
        pass

    def post_pre_train_function(self):
        pass

    def post_round_function(self):
        pass
    
    
    
    
    