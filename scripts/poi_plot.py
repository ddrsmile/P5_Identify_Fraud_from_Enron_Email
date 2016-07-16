"""
@author: Nan-Tsou Liu
created_at: 2016-07-15

Tools for creating the plots of original dataset to get the insight.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
file_path = os.path.dirname(os.path.abspath(__file__))

def concat_dataframe(poi_loss, non_poi_loss, total_loss, normalize=False):

    # concat dataframe with columns
    df = pd.concat([poi_loss, non_poi_loss, total_loss], axis=1)

    # remove poi column because everyone is marked
    df = df.drop('poi')

    # rename the column
    df.columns = ['poi_loss', 'non_poi_loss', 'total_loss']
    df.reset_index(level=0, inplace=True)
    if normalize:
        df['poi_loss'] = df['poi_loss'] / 18
        df['non_poi_loss'] = df['non_poi_loss'] / (143 - 18)
        df['total_loss'] = df['total_loss'] / 143
    print df

    return df


def create_plot(poi_loss, non_poi_loss, total_loss, normalize=False):

    # define figure file name
    title = 'normalized_feature_loss_count' if normalize else 'actual_feature_loss_count'

    # get contacted dataframe of feature loss information
    df = concat_dataframe(poi_loss, non_poi_loss, total_loss, normalize)

    # define the position in x-axis
    pos = list(range(len(df.index)))
    width = 0.25

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(20, 10))

    # Create a bar with poi data,
    # in position pos,
    plt.bar(pos,
            #using df['pre_score'] data,
            df['poi_loss'],
            # of width
            width,
            # with alpha 0.5
            alpha=0.5,
            # with color
            color='#EE3224',
            # with label the first value in first_name
            #label=df['first_name'][0]
            )

    # Create a bar with non-poi data,
    # in position pos + some width buffer,
    plt.bar([p + width for p in pos],
            #using df['mid_score'] data,
            df['non_poi_loss'],
            # of width
            width,
            # with alpha 0.5
            alpha=0.5,
            # with color
            color='#F78F1E',
            # with label the second value in first_name
            # label=df['first_name'][1]
            )

    # Create a bar with total data,
    # in position pos + some width buffer,
    plt.bar([p + width*2 for p in pos],
            #using df['post_score'] data,
            df['total_loss'],
            # of width
            width,
            # with alpha 0.5
            alpha=0.5,
            # with color
            color='#FFC222',
            # with label the third value in first_name
            # label=df['first_name'][2]
            )

    # Set the y axis label
    ax.set_ylabel('loss count')

    # Set the chart's title
    fig_title = ' '.join(title.title().split('_'))
    ax.set_title('{}'.format(fig_title))

    # Set the position of the x ticks
    ax.set_xticks([p + 1.5 * width for p in pos])

    # Set the labels for the x ticks
    ax.set_xticklabels(df['index'], rotation=70, fontsize=12)

    # Setting the x-axis and y-axis limits
    plt.xlim(min(pos)-width, max(pos)+width*4)
    if normalize:
        plt.ylim([0, 1])
    else:
        plt.ylim([0, 150])

    # Adding the legend and showing the plot
    plt.legend(['poi_loss', 'non_poi_loss', 'total_loss'], loc='upper left')
    #fig.tight_layout()
    #plt.show()
    fig.savefig(os.path.join(file_path, '{}.png'.format(title)), bbox_inches='tight')

