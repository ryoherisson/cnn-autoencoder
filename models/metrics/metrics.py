from pathlib import Path
import csv
from logging import getLogger

import numpy as np
import pandas as pd

import torch

logger = getLogger(__name__)
pd.set_option('display.unicode.east_asian_width', True)

class Metrics(object):
    def __init__(self, n_classes, classes, writer, metrics_dir, plot_confusion_matrix, epsilon=1e-12):
        self.n_classes = n_classes
        self.classes = classes
        self.init_cmx()
        self.epsilon = epsilon
        self.loss = 0
        self.writer = writer
        self.metrics_dir = metrics_dir

        # function usage: self.plot_confusion_matrix()
        self.plot_confusion_matrix = plot_confusion_matrix

    def update(self, preds, targets, loss, accuracy):
        stacked = torch.stack((targets, preds), dim=1)

        for p in stacked:
            tl, pl = p.tolist()
            self.__cmx[tl, pl] = self.__cmx[tl, pl] + 1

        self.loss = loss
        self.accuracy = accuracy

    def calc_metrics(self, epoch, mode='train', inference=False):
        tp = torch.diag(self.__cmx).to(torch.float32)
        fp = (self.__cmx.sum(axis=1) - torch.diag(self.__cmx)).to(torch.float32)
        fn = (self.__cmx.sum(axis=0) - torch.diag(self.__cmx)).to(torch.float32)

        self.precision = tp / (tp + fp + self.epsilon)
        self.recall = tp / (tp + fn + self.epsilon)
        self.f1score = tp / (tp + 0.5 * (fp + fn) + self.epsilon) # micro f1score

        self.logging(epoch, mode)
        self.save_csv(epoch, mode)

        # if inference is True, plot confusion matrix
        if inference:
            self.plot_confusion_matrix(self.__cmx.clone().numpy(), self.classes, self.metrics_dir)

    def init_cmx(self):
        """Initialize Confusion Matrix tensor with shape (n_classes, n_classes)
        """
        self.__cmx = torch.zeros(self.n_classes, self.n_classes, dtype=torch.int64)

    def logging(self, epoch, mode):
        logger.info(f'{mode} metrics...')
        logger.info(f'loss:         {self.loss}')
        logger.info(f'accuracy:     {self.accuracy}')

        df = pd.DataFrame(index=self.classes)
        df['precision'] = self.precision.tolist()
        df['recall'] = self.recall.tolist()
        df['f1score'] = self.f1score.tolist()

        logger.info(f'\nmetrics values per classes: \n{df}\n')

        logger.info(f'precision:    {self.precision.mean()}')
        logger.info(f'recall:       {self.recall.mean()}')
        logger.info(f'mean_f1score: {self.f1score.mean()}\n') # micro mean f1score

        # Change mode from 'test' to 'val' to change the display order from left to right to train and test.
        mode = 'val' if mode == 'test' else mode

        self.writer.add_scalar(f'loss/{mode}', self.loss, epoch)
        self.writer.add_scalar(f'accuracy/{mode}', self.accuracy, epoch)
        self.writer.add_scalar(f'mean_f1score/{mode}', self.f1score.mean(), epoch)
        self.writer.add_scalar(f'precision/{mode}', self.precision.mean(), epoch)
        self.writer.add_scalar(f'recall/{mode}', self.recall.mean(), epoch)

    def save_csv(self, epoch, mode):
        csv_path = self.metrics_dir / f'{mode}_metrics.csv'

        if not csv_path.exists():
            with open(csv_path, 'w') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow(['epoch', f'{mode} loss', f'{mode} accuracy',
                                    f'{mode} precision', f'{mode} recall', f'{mode} micro f1score'])

        with open(csv_path, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, self.loss, self.accuracy, 
                                self.precision.mean().item(), self.recall.mean().item(), self.f1score.mean().item()])