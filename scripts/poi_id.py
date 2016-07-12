#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from tools.feature_format import featureFormat, targetFeatureSplit
from sklearn.grid_search import GridSearchCV
from poi_validate import *
from poi_data import *
from poi_add_features import *
from poi_pipeline import *
import pandas as pd
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
# This person is removed because there is no any data of this person.
data_dict.pop('LOCKHART EUGENE E')


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
data_dict = add_features(data_dict)
data_dict = fill_zeros(data_dict)

my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

if __name__ == "__main__":

    ### Task 4: Try a varity of classifiers
    ### Please name your classifier clf for easy export below.
    ### Note that if you want to do PCA or other multi-stage operations,
    ### you'll need to use Pipelines. For more info:
    ### http://scikit-learn.org/stable/modules/pipeline.html

    df = pd.DataFrame.from_dict(data_dict, orient='index')
    features, labels = features_split_pandas(df)

    sk_fold = StratifiedShuffleSplit(labels, n_iter=1000, test_size=0.1)

    # Provided to give you a starting point. Try a variety of classifiers.
    #pipeline = get_LogReg_pipeline()
    #params = get_LogReg_params()

    pipeline = get_SVC_pipeline()
    params = get_SVC_params()

    # scoring_metric: average_precision, roc_auc, f1, recall, precision
    scoring_metric = 'precision'
    grid_searcher = GridSearchCV(pipeline, param_grid=params, cv=sk_fold,
                                 n_jobs=-1, scoring=scoring_metric, verbose=0)


    grid_searcher.fit(features, y=labels)
    mask = grid_searcher.best_estimator_.named_steps['selection'].get_support()
    top_features = [x for (x, boolean) in zip(features, mask) if boolean]
    n_pca_components = grid_searcher.best_estimator_.named_steps['reducer'].n_components_

    print "Cross-validated {0} score: {1}".format(scoring_metric, grid_searcher.best_score_)
    print "{0} features selected".format(len(top_features))
    print "Reduced to {0} PCA components".format(n_pca_components)
    ###################
    # Print the parameters used in the model selected from grid search
    print "Params: ", grid_searcher.best_params_
    ###################

    clf = grid_searcher.best_estimator_
    validate(clf, features, labels)
    ### Task 5: Tune your classifier to achieve better than .3 precision and recall
    ### using our testing script. Check the tester.py script in the final project
    ### folder for details on the evaluation method, especially the test_classifier
    ### function. Because of the small size of the dataset, the script uses
    ### stratified shuffle split cross validation. For more info:
    ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

    # Example starting point. Try investigating other evaluation techniques!


    ### Task 6: Dump your classifier, dataset, and features_list so anyone can
    ### check your results. You do not need to change anything below, but make sure
    ### that the version of poi_id.py that you submit can be run on its own and
    ### generates the necessary .pkl files for validating your results.

    #dump_classifier_and_data(clf, my_dataset, features_list)
