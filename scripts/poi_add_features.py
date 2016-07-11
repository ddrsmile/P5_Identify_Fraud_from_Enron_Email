"""
@author: Nan-Tsou Liu
created_at: 2016-07-10

Api for adding ratios as the features to be used in creating fraud person-of-interest (POI) prediction model.

"""

def add_features(data_dict):
    data_dict = add_financial_ratois(data_dict)
    data_dict = add_email_ratios(data_dict)
    return data_dict


def add_financial_ratois(data_dict):
    financial_features = ['salary',
                          'deferral_payments',
                          'bonus',
                          'expenses',
                          'loan_advances',
                          'other',
                          'director_fees',
                          'deferred_income',
                          'long_term_incentive',
                          'exercised_stock_options',
                          'restricted_stock',
                          'restricted_stock_deferred']

    for person in data_dict:
        try:
            data_dict[person]['total_financial'] = data_dict[person]['total_payments'] + data_dict[person]['total_stock_value']
            for key, val in data_dict[person]:
                if key in financial_features:
                    if data_dict[person]['total_financial'] == 0:
                        data_dict[person]['{}_ratio'.format(key)] = 0
                    else:
                        data_dict[person]['{}_ratio'.format(key)] = val / data_dict[person]['total_financial']

        except:
            data_dict[person]['total_financial'] = 'NaN'

    return data_dict


def add_email_ratios(data_dict):
    for person in data_dict:
        try:
            total_messages = data_dict[person]['from_messages'] + data_dict[person]['to_messages']
            poi_related_messages = data_dict[person]['from_poi_to_this_person'] + data_dict[person]['from_this_person_to_poi'] + data_dict[person]['shared_receipt_with_poi']
            poi_ratio = 1.0 * poi_related_messages / total_messages
            data_dict[person]['poi_ratio_messages'] = poi_ratio
        except:
            data_dict[person]['poi_ratio_messages'] = 'NaN'

    return data_dict
