"""Useful functions for logging."""
# Authors: Christian Ferreyra, chrisferreyra13@gmail.com
# Date: 2022

import logging


def setup_logging(log_filepath):
    """Setup global logging."""
    if not isinstance(log_filepath, str):
        raise TypeError('log filepath should be str')

    # init global logging with defined format
    logging.basicConfig(filename=log_filepath, filemode='w',
                        format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO,
                        datefmt='%d/%m/%Y %I:%M:%S %p')


def setup_logger(name, log_file,
                 log_format=None, datefmt=None, level=logging.DEBUG):
    """Setup an independent logger.
    Parameters
    ----------
    name : str
        Logger name.
    log_file : str
        Filepath for log file.
    log_format : str, optional (default None)
        Logging format
    datefmt : str, optional (default None)
        Datetime format.
    level : logging.level or str, optional (default logging.DEBUG)
        Logging level.
    Returns
    -------
    Logger
        Logger with specific parameters.
    Raises
    ------
    ValueError
        If 'name' or 'log_file' are not provided.
    """

    if not name or not log_file:
        raise ValueError("'name' or 'log_file' not provided")

    handler = logging.FileHandler(log_file)

    if not log_format:
        log_format = '%(asctime)s | %(levelname)s | %(message)s'
    if not datefmt:
        datefmt = '%d/%m/%Y %I:%M:%S %p'

    formatter = logging.Formatter(log_format, datefmt='%d/%m/%Y %I:%M:%S %p')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return
