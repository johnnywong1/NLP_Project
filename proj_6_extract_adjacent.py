import pandas as pd
import re
import os
import logging
import sys

import warnings
warnings.filterwarnings("ignore")

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
    #neg_data["label"] = 0
    #print(neg_data[['data']].tail(1))
    
    # RNV uses a special preprocess step
    print("Preprocessing... 1. split new lines, 2. convert to lowercase, and 3. strip numbers and punct")
    ### 1) remove newlines
    #neg_data['data'] = neg_data['data'].replace('\n', ' ', regex = True)

    ## 2) convert to lowercase
    #neg_data['data'] = neg_data['data'].str.lower()

    # ### 3) remove punct and numbers: https://stackoverflow.com/questions/47947438/preprocessing-string-data-in-pandas-dataframe
    #neg_data["data"] = neg_data.data.apply(lambda x : " ".join(re.findall('[\w]+',x)))
    
    return neg_data

def get_toxic_users(neg_data,min_threshold=5):
    
    toxic_df = neg_data[neg_data['label']==1.0]
    toxic_author = toxic_df['author'].value_counts().to_frame().reset_index()
    toxic_author.rename(columns = {'index':'author','author':'toxic_count'},inplace = True)
    
    top_toxic_author = toxic_author[toxic_author.toxic_count>=min_threshold]
    toxic_user_list = top_toxic_author['author'].tolist()
    #toxic_user_list.remove('[deleted]')
    print('extracted toxic users')
    
    return toxic_user_list

def get_adjacent_dataset(user_list,dummy_data,max_threshold=500):
    # if path:
        # neg_data = pd.read_csv(path, usecols=['body'], dtype="string")
    # else:
    pos_data = dummy_data[dummy_data.author.isin(user_list)]
    pos_author = pos_data['author'].value_counts().to_frame().reset_index()
    pos_author.rename(columns = {'index':'author','author':'toxic_count'},inplace = True)
    pos_toxic_author = pos_author[pos_author.toxic_count<=max_threshold]
    toxic_user_list = pos_toxic_author['author'].tolist()
    pos_data = pos_data[pos_data.author.isin(toxic_user_list)]
    print('extracted comments of toxic users')
    
    neg_data = dummy_data[~dummy_data.author.isin(user_list)]
    print('extracted comments of non-toxic users')
    return pos_data,neg_data


day = sys.argv[1] #first parameter for notebook, for python code it will be sys.argv[1]
min_toxic_comment = int(sys.argv[2]) # second paramter for notebook, for python code it will be sys.argv[2]
max_toxic_comment = int(sys.argv[3]) # second paramter for notebook, for python code it will be sys.argv[2]
#sample_size = int(sys.argv[3]) # third parameter for notebook, for python code it will be sys.argv[3]
#num_curse_words = int(sys.argv[4]) #fourth parameter for notebook, for python code it will be sys.argv[4]
#curse_word_list = ['crap','damn','trash'] #fourth parameter for notebook, will comment it out in python code

# following block will be used in python code
#curse_word_list = []

#for i in range(num_curse_words):
#    cur_word = sys.argv[5+i]
#    curse_word_list.append(cur_word)

logfilename = 'log_files/adjacent_'+day+'.log'

# if os.path.isfile(logfilename):
#     os.remove(logfilename)

if not os.path.isdir('log_files/'):
    os.makedirs('log_files/')

logging.basicConfig(filename=logfilename, filemode='w', format='%(message)s', level=logging.INFO)
# Create the logger
# Admin_Client: The name of a logger defined in the config file
mylogger = logging.getLogger('Admin_Client')
mylogger.info('starting log....')

#dummy_df = preprocess_dataset(day)
dummy_df = preprocess_dataset(day)
print(dummy_df.shape)
toxic_users = get_toxic_users(dummy_df,min_threshold=min_toxic_comment)
print("number of toxic users ",len(toxic_users))
print(toxic_users[0:10])

pos_df,neg_df = get_adjacent_dataset(toxic_users,dummy_df,max_threshold=max_toxic_comment)
print(pos_df.shape)
print(neg_df.shape)
pos_df['label'] = 1.0
neg_df['label'] = 0.0

adjacent_loc = "/bigtemp/rm5tx/nlp_project/adjacent_data/"

if not os.path.isdir(adjacent_loc):
    os.makedirs(adjacent_loc)

pos_df.to_csv(adjacent_loc+"adjacent_"+day+"_min_"+str(min_toxic_comment)+"_max_"+str(max_toxic_comment)+"_positive.csv")
print('written positive dataset')
neg_df.to_csv(adjacent_loc+"adjacent_"+day+"_min_"+str(min_toxic_comment)+"_max_"+str(max_toxic_comment)+"_negative.csv")