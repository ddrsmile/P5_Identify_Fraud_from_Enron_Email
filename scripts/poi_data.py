"""
@author: Nan-Tsou Liu
created_at: 2016-07-10

Api for clean the dataset to be used in creating fraud person-of-interest (POI) prediction model.

"""

def fill_zeros(data_dict):
    target = ['NaN', 'NaNNaN']
    for person in data_dict:
        data_dict[person].pop('email_address')
        for key, val in data_dict[person].items():
            if val in target:
                data_dict[person][key] = 0

    return data_dict


def fix_records(data_dict):
    data_dict['BELFER ROBERT'] = {'bonus': 'NaN',
     'deferral_payments': 'NaN',
     'deferred_income': -102500,
     'director_fees': 102500,
     'email_address': 'NaN',
     'exercised_stock_options': 'NaN',
     'expenses': 3285,
     'from_messages': 'NaN',
     'from_poi_to_this_person': 'NaN',
     'from_this_person_to_poi': 'NaN',
     'loan_advances': 'NaN',
     'long_term_incentive': 'NaN',
     'other': 'NaN',
     'poi': False,
     'restricted_stock': -44093,
     'restricted_stock_deferred': 44093,
     'salary': 'NaN',
     'shared_receipt_with_poi': 'NaN',
     'to_messages': 'NaN',
     'total_payments': 3285,
     'total_stock_value': 'NaN'}
    data_dict['BHATNAGAR SANJAY'] = {'bonus': 'NaN',
     'deferral_payments': 'NaN',
     'deferred_income': 'NaN',
     'director_fees': 'NaN',
     'email_address': 'sanjay.bhatnagar@enron.com',
     'exercised_stock_options': 15456290,
     'expenses': 137864,
     'from_messages': 29,
     'from_poi_to_this_person': 0,
     'from_this_person_to_poi': 1,
     'loan_advances': 'NaN',
     'long_term_incentive': 'NaN',
     'other': 'NaN',
     'poi': False,
     'restricted_stock': 2604490,
     'restricted_stock_deferred': -2604490,
     'salary': 'NaN',
     'shared_receipt_with_poi': 463,
     'to_messages': 523,
     'total_payments': 137864,
     'total_stock_value': 15456290}
    return data_dict


def features_split_pandas(df):
    features = df.drop('poi', axis=1).astype(float)
    labels = df['poi']
    labels = labels[features.abs().sum(axis=1) != 0]
    features = features[features.abs().sum(axis=1) != 0]
    return (features, labels)


def combine_to_dict(features = None, labels = None):
    features.insert(0, 'poi', labels)
    data_dict = features.T.to_dict()
    del features
    return data_dict
