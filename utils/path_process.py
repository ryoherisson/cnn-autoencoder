from pathlib import Path

class Paths(object):
    def __init__(self, log_dir):
        # log parent dir
        # e.g. log_dir = './logs/2020-07-26T11:54:29.429655'
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True, parents=True)

        # logfile path
        self.logfile = self.log_dir / 'logfile.log'

        # summary outdir
        self.summary_dir = self.log_dir / 'tensorboard'
        self.summary_dir.mkdir(exist_ok=True)

        # metrics outdir
        self.metrics_dir = self.log_dir / 'metrics'
        self.metrics_dir.mkdir(exist_ok=True)

        # checkpoint outdir
        self.ckpt_dir = self.log_dir / 'checkpoint'
        self.ckpt_dir.mkdir(exist_ok=True)

