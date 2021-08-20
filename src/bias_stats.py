import re
import csv
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

data_path = sys.argv[1]
model_for = sys.argv[2]
word_list = sys.argv[3]

df_word = pd.read_csv(word_list)
noi_wordlist = df_word[df_word.categorization=='harmless-minority'].word.tolist()
oi_wordlist = df_word[df_word.categorization=='offensive-minority-reference'].word.tolist()
oni_wordlist = df_word[df_word.categorization=='offensive-not-minority'].word.tolist()

idtyRe = re.compile(r"\b"+r"\b|\b".join(noi_wordlist)+"\b",re.IGNORECASE)
oiRe = re.compile(r"\b"+r"\b|\b".join(oi_wordlist)+"\b",re.IGNORECASE)
oniRe = re.compile(r"\b"+r"\b|\b".join(oni_wordlist)+"\b",re.IGNORECASE)

"""
This file provides the tools for calculating relevant bias-measuring
statistics. One needs to prepare a .csv file as the data input similar to
the demo file in the data directory.
"""
df = pd.read_csv(data_path)

def select_re_df(df, type=None):
    """
    select relevant dataframe to calculate statistics.
    """

    if type=='aav':
        return df[df.dialect_argmax=='aav']
    elif type=='noi':
        return df[df["tweet"].apply(idtyRe.findall).astype(bool)]
    elif type=='oi':
        return df[df["tweet"].apply(oiRe.findall).astype(bool)]
    elif type=='oni':
        return df[df["tweet"].apply(oniRe.findall).astype(bool)]
    else:
        return df

def get_scores(df, model):
    """
    calculate relevant statistics for measuring bias.
    """
    prediction=df[model].values.tolist()
    prediction = [1 if i>0.5 else 0 for i in prediction]
    labels = df['ND_label'].values.tolist()
    acc = accuracy_score(labels, prediction)
    f1 = f1_score(labels, prediction)
    recall_pos = recall_score(labels, prediction)
    recall_neg = recall_score(labels, prediction, pos_label=0)

    return [acc,f1, 1-recall_neg]

def get_scores_demo(df, model):
    """
    calculate relevant statistics for measuring bias in user-self reported
                  dataset.
    """

    prediction=df[model].values.tolist()
    prediction = [1 if i>0.5 else 0 for i in prediction]
    labels = df['dialect'].values.tolist()
    count_dict = {'aa':[0, 0], 'white':[0, 0], 'all':[0,0], 'multi':[0,0], 'hisp':[0,0], 'other':[0,0], 'asian':[0,0]}
    for i,j in zip(labels, prediction):
        if i not in count_dict:
            i='other'
        count_dict[i][1]+=1
        count_dict['all'][1]+=1
        if j==1:
            count_dict[i][0]+=1
            count_dict['all'][0]+=1
    a = np.array([count_dict[i][0]/(count_dict[i][1]) for i in count_dict])
    return [a[1],a[0],a[0]/a[1], a[0]-a[1]]

