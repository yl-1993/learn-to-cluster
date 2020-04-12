from mmcv.runner import Runner as _Runner


class Runner(_Runner):
    def train_gcnv(self, dataset, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = [None]  # dummy data_loader
        self._max_iters = self._max_epochs
        self.call_hook('before_train_epoch')
        self._loss = 0
        data_batch = dataset
        self.call_hook('before_train_iter')
        _outputs = self.batch_processor(self.model,
                                        data_batch,
                                        train_mode=True,
                                        **kwargs)
        if not isinstance(_outputs, dict):
            raise TypeError('batch_processor() must return a dict')
        if 'log_vars' in _outputs:
            self.log_buffer.update(_outputs['log_vars'],
                                   _outputs['num_samples'])
        self._loss += _outputs['loss']
        self.outputs = {'loss': self._loss}

        self.call_hook('after_train_iter')
        self._loss = 0
        self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1
