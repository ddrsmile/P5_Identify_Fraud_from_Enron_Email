"""
@author: Nan-Tsou Liu
created_at: 2016-07-10

Api for adding ratios as the features to be used in creating fraud person-of-interest (POI) prediction model.
Api to get the KBest result used to compare it with the result of GridSearchCV

"""


def add_features(df):

    df = add_email_ratios(df)
    df = add_financial_ratios(df)

    return df


def add_email_ratios(df):

    total_messages = df['from_messages'] + df['to_messages']
    poi_related_messages = df['from_poi_to_this_person'] + \
                           df['from_this_person_to_poi'] + \
                           df['shared_receipt_with_poi']
    df['poi_ratio_messages'] =  poi_related_messages / total_messages

    return df


    #for person in data_dict:
    #    try:
    #        total_messages = data_dict[person]['from_messages'] + data_dict[person]['to_messages']
    #
    #        poi_related_messages = data_dict[person]['from_poi_to_this_person'] + \
    #                               data_dict[person]['from_this_person_to_poi'] + \
    #                               data_dict[person]['shared_receipt_with_poi']
    #
    #        poi_ratio = 1.0 * poi_related_messages / total_messages
    #        data_dict[person]['poi_ratio_messages'] = poi_ratio
    #
    #    except:
    #        data_dict[person]['poi_ratio_messages'] = 'NaN'
    #
    #return data_dict, ['poi_ratio_messages']


def add_financial_ratios(df):

    df = create_payments_ratio(df)
    df = create_stock_value_ratio(df)
    df = create_overall_financial_ratio(df, saperate=False)

    return df


    #financial_features = ['salary',
    #                      'deferral_payments',
    #                      'bonus',
    #                      'expenses',
    #                      'loan_advances',
    #                      'other',
    #                      'director_fees',
    #                      'deferred_income',
    #                      'long_term_incentive',
    #                      'exercised_stock_options',
    #                      'restricted_stock',
    #                      'restricted_stock_deferred']
    #
    #new_financial_features = ['{}_ratio'.format(feature) for feature in financial_features]
    #
    #for person in data_dict:
    #
    #    if data_dict[person]['total_payments'] == 'NaN':
    #        data_dict[person]['total_payments'] = 0
    #
    #    if data_dict[person]['total_stock_value'] == 'NaN':
    #        data_dict[person]['total_stock_value'] = 0
    #
    #    data_dict[person]['total_financial'] = data_dict[person]['total_payments'] + \
    #                                           data_dict[person]['total_stock_value']
    #
    #    for key, val in data_dict[person].items():
    #        if key in financial_features:
    #            new_feature = '{}_ratio'.format(key)
    #
    #            if val == 'NaN' or data_dict[person]['total_financial'] == 0:
    #                data_dict[person][new_feature] = 0.0
    #            else:
    #                data_dict[person][new_feature] = 1.0 * val / data_dict[person]['total_financial']
    #
    #return data_dict, new_financial_features


def create_payments_ratio(df):

    # the features related to payments
    payment_features = ['salary', 'deferral_payments', 'bonus',
                        'expenses','loan_advances', 'other',
                        'director_fees','deferred_income', 'long_term_incentive']

    for feature in payment_features:
        df['{0}_payment_ratio'.format(feature)] = df[feature] / df['total_payments']

    return df


def create_stock_value_ratio(df):

    # the features related to payments
    stock_value_features = ['exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred']

    for feature in stock_value_features:
        df['{0}_stock_ratio'.format(feature)] = df[feature] / df['total_stock_value']

    return df


def create_overall_financial_ratio(df, saperate=False):
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

    df['total_financial'] = df['total_payments'] + df['total_stock_value']

    if saperate:
        df['overall_pay_ratio'] = df['total_payments'] / df['total_financial']
        df['overall_stock_ratio'] = df['total_stock_value'] / df['total_financial']
    else:
        for feature in financial_features:
            df['{}_ratio'.format(feature)] = df[feature] / df['total_financial']



    return df
