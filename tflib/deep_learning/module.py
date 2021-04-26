import uuid

import tensorflow as tf


class Module:

    __existing_modules = {}

    @classmethod
    def existing_modules(cls):
        return cls.__existing_modules

    def __init__(self, scope=None):
        self._scope = (type(self).__name__ + '_' + uuid.uuid4().hex[:4]) if scope in [None, ''] else scope
        assert self._scope not in Module.__existing_modules, 'Scope name "%s" has been used!' % self._scope
        Module.__existing_modules[self._scope] = type(self)

        self._variables = []
        self._trainable_variables = []
        self._reg_losses = []
        self._reg_loss = None
        self._is_built = False

    def __call__(self, *args, **kwargs):
        """Build graph implemented in `call` function of a subclass.

        Notice
        -------
        - Variables are created only at the first call and will be reused afterwards.
        - Do not run the graph in any variable scope.

        """
        assert tf.get_variable_scope().name == '', 'Do not use the module in any variable scope!'

        with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):
            results = self.call(*args, **kwargs)

        if not self._is_built:
            self._variables = tf.global_variables(self._scope)
            self._trainable_variables = tf.trainable_variables(self._scope)
            self._reg_losses = tf.losses.get_regularization_losses(self._scope)
            self._reg_loss = tf.losses.get_regularization_loss(self._scope)
            self._is_built = True

        return results

    def call(self, *args, **kwargs):
        raise NotImplementedError

    def clone_from_vars(self, src_vars, var_type='all'):
        assert var_type in ['all', 'trainable', 'nontrainable']
        assert self._is_built, 'The target module should be already built (called at least once)!'

        trg_vars = {'all': self.variables, 'trainable': self.trainable_variables, 'nontrainable': self.nontrainable_variables}[var_type]
        clone_ops = []
        for trg_var, src_var in zip(trg_vars, src_vars):
            clone_ops.append(trg_var.assign(src_var))

        return clone_ops

    def clone_from_module(self, src_module, var_type='all'):
        assert var_type in ['all', 'trainable', 'nontrainable']
        assert type(self) == type(src_module), 'The types of the target module and the source module are inconsistent!'
        assert self._is_built, 'The target module should be already built (called at least once)!'
        assert src_module._is_built, 'The source module should be already built (called at least once)!'

        src_vars = {'all': src_module.variables, 'trainable': src_module.trainable_variables, 'nontrainable': src_module.nontrainable_variables}[var_type]
        clone_ops = self.clone_from_vars(src_vars, var_type=var_type)

        return clone_ops

    @property
    def scope(self):
        return self._scope

    @property
    def variables(self):
        return self._variables

    @property
    def trainable_variables(self):
        return self._trainable_variables

    @property
    def nontrainable_variables(self):
        return [var for var in self.variables if var not in self.trainable_variables]

    @property
    def reg_losses(self):
        return self._reg_losses

    @property
    def reg_loss(self):
        return self._reg_loss
