"""
@author: Nan-Tsou Liu
created_at: 2016-07-10

Api for validating the fraud person-of-interest (POI) prediction model with sk-learn modules.
"""

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
import sys


def validate(clf, features, labels, n_iters=1000, test_size=0.1):

    sk_fold = StratifiedShuffleSplit(labels, n_iter=n_iters, test_size=test_size)

    f1 = []
    recall = []
    precision = []

    for i, all_index in enumerate(sk_fold):

        train_index = all_index[0]
        test_index = all_index[1]

        features_train = features.irow(train_index)
        labels_train = labels[train_index]

        features_test = features.irow(test_index)
        labels_test = labels[test_index]

        clf.fit(features_train, labels_train)

        predictions = clf.predict(features_test)
        f1.append(f1_score(labels_test, predictions))
        precision.append(precision_score(labels_test, predictions))
        recall.append(recall_score(labels_test, predictions))

        if i % round(n_iters / 10) == 0:

            sys.stdout.write('{0}%--> '.format(float(i) / n_iters * 100))
            sys.stdout.flush()

    print 'Done!'
    print ''
    print 'F1 Avg: ', sum(f1) / n_iters
    print 'Precision Avg: ', sum(precision) / n_iters
    print 'Recall Avg: ', sum(recall) / n_iters
