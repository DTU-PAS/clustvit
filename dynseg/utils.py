from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class DisableAuxLossHook(Hook):
    def __init__(self, disable_after_iter=1000):
        self.disable_after_iter = disable_after_iter

    def before_train_iter(self, runner, batch_idx, data_batch):
        if runner.iter == self.disable_after_iter:
            runner.model.auxiliary_head.loss_decode[0].loss_weight = 0.0
            runner.logger.info(f"Auxiliary loss disabled at iter {runner.iter}")
