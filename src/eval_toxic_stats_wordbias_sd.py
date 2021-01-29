import csv
import sys
import re
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

data_path = sys.argv[1]
model_for = sys.argv[2]


'''
count = 0
with open(data_path, 'r', encoding='utf-8') as f:
    tweets = csv.reader(f, delimiter=",", quotechar='"')
    next(tweets)
    for i in tweets:
print(count)
'''
data_path += '_n_roberta.csv'
df = pd.read_csv(data_path)

df_word = pd.read_csv('./toxic_language/advToxicityFilt_v0.2/word_based_bias_list.csv')
noi_wordlist = df_word[df_word.categorization=='harmless-minority'].word.tolist()
oi_wordlist = df_word[df_word.categorization=='offensive-minority-reference'].word.tolist()
oni_wordlist = df_word[df_word.categorization=='offensive-not-minority'].word.tolist()

idtyRe = re.compile(r"\b"+r"\b|\b".join(noi_wordlist)+"\b",re.IGNORECASE)
oiRe = re.compile(r"\b"+r"\b|\b".join(oi_wordlist)+"\b",re.IGNORECASE)
oniRe = re.compile(r"\b"+r"\b|\b".join(oni_wordlist)+"\b",re.IGNORECASE)


if data_path.split('/')[-1].startswith('iden'):
    df = df[df["tweet"].apply(idtyRe.findall).astype(bool)]
    print(len(df))

def get_models(mm):
    #n_toxic_model_ambi_s2_lr1e
    a= '_'.join(mm.split('_')[:-2]+['s2','lr1e'])
    b= '_'.join(mm.split('_')[:-2]+['s12','lr1e'])
    c= '_'.join(mm.split('_')[:-2]+['s22','lr1e'])
    return [a,b,c]

def obtain_special_f1(df, model, pos):
    prediction=df[model].values.tolist()
    prediction = [1 if i>0.5 else 0 for i in prediction] 
    labels = df['ND_label'].values.tolist()
    acc = accuracy_score(labels, prediction)
    f1 = f1_score(labels, prediction, pos_label=pos) 
    return f1 

def get_scores(df, model):
    prediction=df[model].values.tolist()
    prediction = [1 if i>0.5 else 0 for i in prediction] 
    labels = df['ND_label'].values.tolist()
    acc = accuracy_score(labels, prediction)
    f1 = f1_score(labels, prediction, pos_label=1) 
    recall_pos = recall_score(labels, prediction)
    recall_neg = recall_score(labels, prediction, pos_label=0)
    #return [acc,f1, recall_pos, recall_neg, 1-recall_neg]
    return [acc, f1, 1-recall_neg]

    '''
    toxic_df = df[df.ND_label==1]
    non_toxic_df = df[df.ND_label==0]
    f1_toxic = obtain_special_f1(toxic_df, model, 1)
    f1_nontoxic = obtain_special_f1(non_toxic_df, model, 0)
    return [f1, f1_toxic, f1_nontoxic]
    '''
def get_scores_sd(df, mode=None):
    print(len(df))
    score_matrix = []
    for model in get_models(model_for):
        score_matrix.append(get_scores(df, model))
    score_matrix = np.array(score_matrix)
    #print('mean and variance')
    #print(score_matrix)
    avg = (np.mean(score_matrix, axis=0))
    std = (np.std(score_matrix, axis=0))
    '''
    if mode=='test':
        avg = (np.max(score_matrix, axis=0))
        std = (np.max(score_matrix, axis=0))
    else:
        avg = (np.min(score_matrix, axis=0))
        std = (np.min(score_matrix, axis=0))
    '''
    if mode:
        acc = avg[0]
        acc_std = std[0]
        #return f'{100*acc:.2f} & {100*avg[1]:.2f} & {100*avg[2]:.2f}'
        #return f'{100*acc:.2f}$_{{{100*acc_std:.1f}}}$ & {100*avg[1]:.2f}$_{{{100*std[1]:.1f}}}$'
        return f'{100*acc:.2f}$_{{{100*acc_std:.1f}}}$ & {100*avg[1]:.2f}$_{{{100*std[1]:.1f}}}$ & {100*avg[2]:.2f}$_{{{100*std[2]:.1f}}}$'
    else:
        fpr = avg[2]
        fpr_std = std[2]
    return f'{100*avg[1]:.2f}$_{{{100*std[1]:.1f}}}$ & {100*fpr:.2f}$_{{{100*fpr_std:.1f}}}$' 
    #return f'{100*avg[1]:.2f} & {100*fpr:.2f}' 
# Use the n to represent the new file below
if data_path.split('/')[-1]=='twitter_user_demo_n_roberta.csv':
    prediction=df[model_for].values.tolist()
    prediction = [1 if i>0.5 else 0 for i in prediction]
    labels = df['dialect'].values.tolist()
    count_dict = {'aa':[0, 0], 'white':[0, 0], 'all':[0,0], 'multi':[0,0], 'hisp':[0,0], 'other':[0,0], 'asian':[0,0]}
    for i,j in zip(labels, prediction):
        if i not in count_dict:
            i='other'
        count_dict[i][1]+=1
        count_dict['all'][1]+=1
        #print(j)
        if j==1:
            count_dict[i][0]+=1
            count_dict['all'][0]+=1
    # Count the dialects number
    a = np.array([count_dict[i][0]/(count_dict[i][1]-count_dict[i][0]) for i in count_dict])
    print(a[1],a[0],a[0]/a[1], a[0]-a[1])
else:
    a = get_scores_sd(df,'test')
    if data_path.split('/')[-1].startswith('ND'):
        a= a+ ' & '+get_scores_sd(df[df["tweet"].apply(idtyRe.findall).astype(bool)])
        a= a+ ' & '+get_scores_sd(df[df["tweet"].apply(oiRe.findall).astype(bool)])
        a= a+ ' & '+get_scores_sd(df[df["tweet"].apply(oniRe.findall).astype(bool)])
    if data_path.split('/')[-1].startswith('identity'):
        a= a+ ' & '+get_scores_sd(df)
    print(a)
    #get_scores(df[df.dialect_argmax!='aav'])
