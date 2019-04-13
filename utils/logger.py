#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging


def create_logger(name='global_logger', log_file=None):
    """ use different log level for file and stream
    """
    logger = logging.getLogger(name)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    logger.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)

    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

    return logger


if __name__ == '__main__':
    logger = create_logger('test')
    logger = create_logger('test', 'log.txt')
    logger.info('output to file and stream')
    logger.debug('output to file')
