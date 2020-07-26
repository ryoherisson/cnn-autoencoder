from pathlib import Path

import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cmx, classes, metrics_dir, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cmx = cmx.astype('float') / cmx.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix\n')
    else:
        print('Confusion matrix, without normalization\n')

    plt.imshow(cmx, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylim(len(classes) - 0.5, -0.5)

    fmt = '.2f' if normalize else 'd'
    thresh = cmx.max() / 2.
    for i, j in itertools.product(range(cmx.shape[0]), range(cmx.shape[1])):
        plt.text(j, i, format(cmx[i, j], fmt), horizontalalignment='center', color='white' if cmx[i, j] > thresh else 'black')
    
    plt.tight_layout()
    plt.ylabel('True lable')
    plt.xlabel('Predicted label')

    cmx_path = metrics_dir / 'test_cmx.png'
    plt.savefig(cmx_path)

    plt.close()

    