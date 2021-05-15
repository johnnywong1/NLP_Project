import pandas as pd
# import re
import os
# import logging
import sys


#model_1_output_file = '/bigtemp/rm5tx/nlp_project/2016-07_base_model.csv'
#model_2_output_file = '/bigtemp/rm5tx/nlp_project/2016-07_adj_model.csv'

model_1_output_file = '/bigtemp/rm5tx/nlp_project/2016-07_all_predicted_base_model_2.csv'
model_2_output_file = '/bigtemp/rm5tx/nlp_project/2016-07_all_predicted_adj_model_2.csv'

df1 = pd.read_csv(model_1_output_file,nrows=100000)
df1 = df1.dropna(subset=['id'])
print(df1.shape)
print(df1[df1.label==1.0][['author','data','label']].head(20))
print(df1[df1.label==1.0][['author','data','label']].shape)
print(df1['id'].nunique())
print(df1['id'].value_counts())

df2 = pd.read_csv(model_2_output_file,nrows=100000)
df2 = df2.dropna(subset=['id'])
print(df2.shape)
print(df2[df2.label==1.0][['author','data','label']].head(20))
print(df2[df2.label==1.0][['author','data','label']].shape)
print(df2['id'].nunique())