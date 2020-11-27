import pandas as pd
import numpy as np
import re
import csv
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import sys

if len(sys.argv)>1:
    mode = sys.argv[1]
else:
    mode = 'all'

training_path = '../toxic_language/advToxicityFilt_v0.2/ND_founta_trn_dial_pAPI.csv' 
print(training_path.split('/')[-2])
eval_path = '../toxic_language/advToxicityFilt_v0.2/ND_founta_tst_dial_pAPI.csv'

#toxic_identity_list = ['blind','young','asian','male','muslim','latino','female','gay','queer','islam' ,'lesbian' ,'latina'] 

df = pd.read_csv('word_based_bias_list.csv')
#offensive-minority-reference offensive-not-minority harmless-minority  
df = df[df.categorization=='harmless-minority']
b = len(df)
word_list = df.word.tolist()


def get_features(df):
    idtyRe = re.compile(r"\b"+r"\b|\b".join(word_list)+"\b",re.IGNORECASE) 
    word_bias = df["tweet"].apply(idtyRe.findall)
    return [' '.join(i) for i in word_bias.tolist()]
    

# Read in data
data = pd.read_csv(training_path)
# Stats:
oiRe = re.compile(r"\b"+r"\b|\b".join(word_list)+"\b",re.IGNORECASE)
df_a = data[data["tweet"].apply(oiRe.findall).astype(bool)]
print(len(df_a))
print(len(df_a)/(len(data)))

texts = data['tweet']
y = data['ND_label'].tolist()
F = get_features(data)

# Vectorize the text
vectorizer = CountVectorizer(stop_words='english', min_df=0.0001)
X = vectorizer.fit_transform(F).toarray()

# Train the model
model = LinearSVC(class_weight="balanced", dual=False, tol=1e-2, max_iter=1e5)
cclf = CalibratedClassifierCV(base_estimator=model)
cclf.fit(X, y)

# Save the model
#joblib.dump(vectorizer, saved_path+'/'+'vectorizer.joblib')
#joblib.dump(cclf, saved_path+'/'+'model.joblib')


#vectorizer = joblib.load(pkg_resources.resource_filename('profanity_check', 'data/vectorizer.joblib'))
#model = joblib.load(pkg_resources.resource_filename('profanity_check', 'data/model.joblib'))

# Record the training bias

proba = cclf.predict_proba(X)
print(proba[1])
proba = np.log(proba)
save_file = training_path[:-4]+'_kkk.pkl' 
with open(save_file, 'wb') as handle:
    pickle.dump(proba, handle, protocol=pickle.HIGHEST_PROTOCOL)

prediction = cclf.predict(X)
acc = accuracy_score(y, prediction)
print(f'Training Accuracy score: {acc}')

# Read in eval data
'''
eval_data = pd.read_csv(eval_path)
eval_texts = eval_data['tweet']
eval_y = eval_data['ND_label']
eval_x = vectorizer.transform(get_features(eval_data)).toarray()

proba = cclf.predict_proba(eval_x)
print(proba[1])
proba = np.log(proba)
save_file = eval_path[:-4]+'_ttbias.pkl' 
with open(save_file, 'wb') as handle:
    pickle.dump(proba, handle, protocol=pickle.HIGHEST_PROTOCOL)

prediction = cclf.predict(eval_x)
acc = accuracy_score(eval_y, prediction)
print(f'Eval Accuracy score: {acc}')
'''
