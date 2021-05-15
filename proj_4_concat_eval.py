import pandas as pd

suffix = [0,1,2,3,4]

all_df = []

out_path = '/bigtemp/rm5tx/nlp_project/2016-07_adj_model.csv'

for suf in suffix:
    print(suf)
    pref = '2016-07_all_predicted_adj_model_'
    try:
        cur_df = pd.read_csv('/bigtemp/rm5tx/nlp_project/'+pref+str(suf)+".csv")
    except:
        continue
    all_df.append(cur_df)

big_df = pd.concat(all_df)
#big_df[big_df.label==1.0].head()
big_df.to_csv(out_path)
