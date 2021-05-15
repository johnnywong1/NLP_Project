import pandas as pd
# import re
import os
# import logging
import sys


def preprocess_dataset(day,nrow=-1):
    if nrow==-1:
        neg_data = pd.read_csv("/bigtemp/rm5tx/nlp_project/"+day+"_for_model_2_tuned.csv")
    else:
        neg_data = pd.read_csv("/bigtemp/rm5tx/nlp_project/"+day+"_for_model_2_tuned.csv",nrows=nrow)
    # We want a unify col name for when we concat pos and neg data
    #neg_data.rename(columns={"body":"data"}, inplace=True)
    print("removing irrelevant data")
    neg_data = neg_data.dropna(subset=['author', 'data'])
    neg_data = neg_data[neg_data.author!='[deleted]']
    neg_data = neg_data[neg_data.author!='AutoModerator']

    return neg_data


day = sys.argv[1] #first parameter for notebook, for python code it will be sys.argv[1]
rat = float(sys.argv[2]) # second paramter for notebook, for python code it will be sys.argv[2]
tot = int(sys.argv[3]) # second paramter for notebook, for python code it will be sys.argv[2]

dummy_df = preprocess_dataset(day)

pos_counts = dummy_df['author'][dummy_df['label']==1.0].value_counts().rename("pos")
neg_counts = dummy_df['author'][dummy_df['label']==0.0].value_counts().rename("neg")
counts = pd.concat([pos_counts, neg_counts], axis=1)
counts['ratio'] = counts.pos / (counts.neg + counts.pos)
counts['total'] = counts.pos + counts.neg
counts = counts.dropna()

print("Total Comments from 'toxic users'", counts['total'][(counts['ratio'] >= rat) & (counts['total'] >= tot)].sum())
toxic_users = counts[(counts['ratio'] >= rat) & (counts['total'] >= tot)].index


pos_df = dummy_df[dummy_df.author.isin(toxic_users)]
neg_df = dummy_df[~dummy_df.author.isin(toxic_users)]


# pos_df['label'] = 1
# neg_df['label'] = 0

adjacent_loc = "/bigtemp/rm5tx/nlp_project/adjacent_data/"

if not os.path.isdir(adjacent_loc):
    os.makedirs(adjacent_loc)

pos_df.to_csv(adjacent_loc+"adjacent_"+day+"_rate_"+str(rat)+"_tot_"+str(tot)+"_positive.csv")
print('written positive dataset')
neg_df.to_csv(adjacent_loc+"adjacent_"+day+"_rate_"+str(rat)+"_tot_"+str(tot)+"_negative.csv")