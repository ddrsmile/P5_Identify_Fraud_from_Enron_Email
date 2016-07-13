#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from tools.feature_format import featureFormat, targetFeatureSplit
from sklearn.grid_search import GridSearchCV
from poi_pipeline import *
from poi_validate import *
from poi_data import *
from poi_add_features import *
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

labels = ['poi']
email_features_list = [
    'from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'shared_receipt_with_poi',
    'to_messages',
    ]
financial_features_list = [
    'bonus',
    'deferral_payments',
    'deferred_income',
    'director_fees',
    'exercised_stock_options',
    'expenses',
    'loan_advances',
    'long_term_incentive',
    'other',
    'restricted_stock',
    'restricted_stock_deferred',
    'salary',
    'total_payments',
    'total_stock_value',
]

features_list = labels + email_features_list + financial_features_list

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
# fix the not consistent data
data_dict = fix_records(data_dict)

# add new feature dataset and get the list of added feature
data_dict, new_features = add_features(data_dict)
features_list += new_features

# cleaning data, replace 'NaN' with 0
data_dict = fill_zeros(data_dict)

# assign data_dict to my_dataset for dumping out
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)


if __name__ == "__main__":

    ### Task 4: Try a varity of classifiers
    ### Please name your classifier clf for easy export below.
    ### Note that if you want to do PCA or other multi-stage operations,
    ### you'll need to use Pipelines. For more info:
    ### http://scikit-learn.org/stable/modules/pipeline.html

    # generate training and testing dataset
    sk_fold = StratifiedShuffleSplit(labels, n_iter=1000, test_size=0.1)

    # Provided to give you a starting point. Try a variety of classifiers.
    # pipeline = get_LogReg_pipeline()
    # params = get_LogReg_params()

    # pipeline = get_SVC_pipeline()
    # params = get_SVC_params()

    # pipeline = get_LSVC_pipeline()
    # params = get_LSVC_params()

    pipeline = get_DTree_pipeline()
    params = get_DTree_params()

    # determine the score to tune the parameters
    scoring_metric = 'precision'

    # run grid search to tune the parameters
    grid_searcher = GridSearchCV(pipeline, param_grid=params, cv=sk_fold,
                                 n_jobs=-1, scoring=scoring_metric, verbose=0)

    grid_searcher.fit(features, labels)

    # extract the parameters for the analysis
    # k-best features
    ## get list of selected features
    mask = grid_searcher.best_estimator_.named_steps['selection'].get_support()

    ## get scores of each features
    k_score = grid_searcher.best_estimator_.named_steps['selection'].scores_

    ## sort the list of features with scores in decendent order
    top_features = [x for (x, boolean) in zip(features_list[1:], mask) if boolean]
    top_score = [x for (x, boolean) in zip(k_score, mask) if boolean]
    sorted_top_features = list(reversed(sorted(zip(top_features, top_score), key=lambda x: x[1])))

    # the number of pca features
    n_pca_components = grid_searcher.best_estimator_.named_steps['reducer'].n_components_

    # the dict of best parameters
    best_params = grid_searcher.best_params_

    # the best score of grid search
    best_score = grid_searcher.best_score_

    # print the results of parameters tuning
    print "Cross-validated {0} score: {1}".format(scoring_metric, best_score)

    # print number of k-best features and the score of each features
    print "{0} features selected".format(len(top_features))
    for feature in sorted_top_features:
        print feature[0], ": ", feature[1]

    # print number of PCA components
    print "Reduced to {0} PCA components".format(n_pca_components)

    # print the parameters used in the model selected from grid search
    print "Params: ", best_params

    ### Task 5: Tune your classifier to achieve better than .3 precision and recall
    ### using our testing script. Check the tester.py script in the final project
    ### folder for details on the evaluation method, especially the test_classifier
    ### function. Because of the small size of the dataset, the script uses
    ### stratified shuffle split cross validation. For more info:
    ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

    clf = grid_searcher.best_estimator_
    validate(clf, data_dict)

    ### Task 6: Dump your classifier, dataset, and features_list so anyone can
    ### check your results. You do not need to change anything below, but make sure
    ### that the version of poi_id.py that you submit can be run on its own and
    ### generates the necessary .pkl files for validating your results.

    #dump_classifier_and_data(clf, my_dataset, features_list)
