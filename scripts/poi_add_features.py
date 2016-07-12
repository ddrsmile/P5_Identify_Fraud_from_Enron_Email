"""
@author: Nan-Tsou Liu
created_at: 2016-07-10

Api for adding ratios as the features to be used in creating fraud person-of-interest (POI) prediction model.

"""


def add_features(data_dict):

    data_dict, new_email_features = add_email_ratios(data_dict)
    data_dict, new_financial_features = add_financial_ratios(data_dict)

    return data_dict, new_financial_features + new_email_features


def add_email_ratios(data_dict):

    for person in data_dict:
        try:
            total_messages = data_dict[person]['from_messages'] + data_dict[person]['to_messages']

            poi_related_messages = data_dict[person]['from_poi_to_this_person'] + \
                                   data_dict[person]['from_this_person_to_poi'] + \
                                   data_dict[person]['shared_receipt_with_poi']

            poi_ratio = 1.0 * poi_related_messages / total_messages
            data_dict[person]['poi_ratio_messages'] = poi_ratio

        except:
            data_dict[person]['poi_ratio_messages'] = 'NaN'

    return data_dict, ['poi_ratio_messages']


def add_financial_ratios(data_dict):
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

    new_financial_features = ['{}_ratio'.format(feature) for feature in financial_features]

    for person in data_dict:
        for key, val in data_dict[person].items():
            if key in financial_features:
                new_feature = '{}_ratio'.format(key)
                try:
                    data_dict[person]['total_financial'] = data_dict[person]['total_payments'] + \
                                                           data_dict[person]['total_stock_value']

                    if data_dict[person]['total_financial'] == 0:
                        data_dict[person][new_feature] = 0
                    else:
                        data_dict[person][new_feature] = val / data_dict[person]['total_financial']

                except:

                    data_dict[person]['total_financial'] = 'NaN'
                    data_dict[person][new_feature] = 'NaN'

    return data_dict, new_financial_features
