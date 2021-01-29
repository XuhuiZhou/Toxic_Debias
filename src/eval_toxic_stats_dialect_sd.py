import csv
import sys
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

def get_models(mm):
    #n_toxic_model_ambi_s2_lr1e
    a= '_'.join(mm.split('_')[:-2]+['s2','lr1e'])
    b= '_'.join(mm.split('_')[:-2]+['s12', 'lr1e'])
    #b= '_'.join(mm.split('_')[1:-2]+['12'])
    c= '_'.join(mm.split('_')[:-2]+['s22','lr1e'])
    return [a,b,c]

def get_scores(df, model):
    prediction=df[model].values.tolist()
    prediction = [1 if i>0.5 else 0 for i in prediction] 
    labels = df['ND_label'].values.tolist()
    acc = accuracy_score(labels, prediction)
    f1 = f1_score(labels, prediction) 
    recall_pos = recall_score(labels, prediction)
    recall_neg = recall_score(labels, prediction, pos_label=0)

    #return [acc,f1, recall_pos, recall_neg, 1-recall_neg]
    return [acc,f1, 1-recall_neg]

def get_scores_demo(df, model):
    prediction=df[model].values.tolist()
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
    a = np.array([count_dict[i][0]/(count_dict[i][1]) for i in count_dict]) 
    return [a[1],a[0],a[0]/a[1], a[0]-a[1]]


def get_scores_sd(df,mode=None):
    score_matrix = []
    for model in get_models(model_for):
        if mode=='user':
            score_matrix.append(get_scores_demo(df, model))
        else:
            score_matrix.append(get_scores(df, model))
    score_matrix = np.array(score_matrix)
    #print('mean and variance')
    avg = (np.mean(score_matrix, axis=0))
    std = (np.std(score_matrix, axis=0))
    if mode=='test':
        acc = avg[0]
        acc_std = std[0]
        return f'{100*acc:.2f}$_{{{100*acc_std:.1f}}}$ & {100*avg[1]:.2f}$_{{{100*std[1]:.1f}}}$'
    elif mode=='user':
        return f'w-tox:{100*avg[0]:.2f}; aa-tox:{100*avg[1]:.2f}; delta:{avg[2]:.2f};\
                aa/w:{100*avg[3]:.2f}'
    else:
        fpr = avg[2]
        fpr_std = std[2]
    return f'{100*fpr:.2f}$_{{{100*fpr_std:.1f}}}$' 


# Use the n to represent the new file below
if data_path.split('/')[-1]=='twitter_user_demo_n_roberta.csv':
    print(get_scores_sd(df,'user'))
else:
    #get_scores_sd(df)
    #get_scores_sd(df[df.dialect_argmax=='aav'])
    #get_scores(df[df.dialect_argmax!='aav'])
    a = get_scores_sd(df,'test')
    a= a+ ' & '+get_scores_sd(df[df.dialect_argmax=='aav'])
    print(a)


