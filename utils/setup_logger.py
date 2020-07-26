import sys
from logging import INFO, DEBUG, FileHandler
from logging import StreamHandler, Formatter, getLogger
from pathlib import Path

def setup_logger(logfile='./logs/logtime.log'):
    logger = getLogger()
    logger.setLevel(INFO)

    # create file handler
    fh = FileHandler(logfile)
    fh.setLevel(INFO)
    fh_formatter = Formatter(fmt='')
    fh.setFormatter(fh_formatter)

    # create console handler
    ch = StreamHandler(stream=sys.stdout)
    ch.setLevel(INFO)
    ch_formatter = Formatter(fmt='')
    ch.setFormatter(ch_formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)