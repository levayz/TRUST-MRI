import logging
import torch.distributed as dist
import re

def create_logger(logging_dir, args=None):
    """
    Create a logger that writes to a log file and stdout.
    """
    try:
        rank_ = dist.get_rank()
    except Exception as err:
        rank_ = 0

    # Convert logging_dir to string if it's a Path object
    logging_dir_str = str(logging_dir)

    # Extract run* pattern from logging_dir
    run_match = logging_dir_str
    run_prefix = f"\033[34m[{run_match}]\033[0m"

    if rank_ == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format=f'{run_prefix}[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def log_args(args, logger):
        message = ''
        message += '\n----------------- Arguments ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        logger.info(message)