import csv
import numpy as np
import pandas as pd
from scipy import stats
from nltk.stem import WordNetLemmatizer
import regex as re
import sys

if len(sys.argv)>1:
    data_path = sys.argv[1]
else:
    data_path = 'aflite/ND_founta_trn_dial_pAPI.csv'
df_tr = pd.read_csv(data_path)

df = pd.read_csv('word_based_bias_list.csv')
OMR = df[df.categorization=='offensive-minority-reference'].word.tolist()
ONM = df[df.categorization=='offensive-not-minority'].word.tolist()
HM = df[df.categorization=='harmless-minority'].word.tolist()

def pearson_tx(word_list):
    idtyRe = re.compile(r"\b"+r"\b|\b".join(word_list)+"\b",re.IGNORECASE)
    df_tr['word'] = df_tr["tweet"].apply(idtyRe.findall).astype(bool)
    nums = len(df_tr[df_tr.word])
    toxic_ratio = (len(df_tr[(df_tr.ND_label==1) & df_tr.word])/len(df_tr[df_tr.word]))
    #return nums, toxic_ratio, stats.pearsonr(df_tr.ND_label, df_tr.word), stats.pearsonr(1-df_tr.ND_label, df_tr.word)
    #return stats.pointbiserialr(df_tr.ND_label, df_tr.word)
    return stats.pearsonr(df_tr.ND_label, df_tr.word)[0]

#print('data length:', len(df_tr))
print(f'{data_path} Pearsonr correlation within each category:')
print(f'noi: {pearson_tx(HM):.4f}; oi: {pearson_tx(OMR):.4f}; oni: \
      {pearson_tx(ONM):.4f}')
print(f'aae: {stats.pearsonr(df_tr.aav, df_tr.ND_label)[0]:.4f}')

