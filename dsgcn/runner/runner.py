import logging
from mmcv.runner import Runner as _Runner


class Runner(_Runner):

    def __init__(self,
                 model,
                 batch_processor,
                 optimizer=None,
                 work_dir=None,
                 log_level=logging.INFO,
                 logger=None,
                 iter_size=1):
        super().__init__(model,
                         batch_processor,
                         optimizer,
                         work_dir,
                         log_level,
                         logger)
        assert isinstance(iter_size, int) and iter_size >= 1
        self.iter_size = iter_size

    def train_iter_size(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(data_loader)
        self.call_hook('before_train_epoch')
        self._loss = 0
        self._iter_size_cnt = 0
        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            _outputs = self.batch_processor(
                self.model, data_batch, train_mode=True, **kwargs)
            if not isinstance(_outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in _outputs:
                self.log_buffer.update(_outputs['log_vars'],
                                       _outputs['num_samples'])
            self._loss += _outputs['loss']
            self._iter_size_cnt += 1
            if (i + 1) % self.iter_size == 0 or i == len(data_loader) - 1:
                self.outputs = {
                    'loss': self._loss / self._iter_size_cnt
                }
                self.call_hook('after_train_iter')
                self._loss = 0
                self._iter_size_cnt = 0
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1
