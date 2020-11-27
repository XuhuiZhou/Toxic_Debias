import pandas as pd
import numpy as np
import re
import csv
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
#from sklearn.externals import joblib
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
#offensive-minority-reference offensive-not-minority harmless-minority  
#df = df[df.categorization=='offensive-not-minority']

def get_features(df):
    white = df["white"].tolist()
    aav = df["aav"].tolist()
    his = df["hispanic"].tolist()
    other = df["other"].tolist()
    return [[float(i[0]), float(i[1]), float(i[2]), float(i[3])] for i in zip(aav, his, other, white)]
    

# Read in data
data = pd.read_csv(training_path)
texts = data['tweet']
y = data['ND_label'].tolist()
X = get_features(data)

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
save_file = training_path[:-4]+'_dddbias.pkl' 
with open(save_file, 'wb') as handle:
    pickle.dump(proba, handle, protocol=pickle.HIGHEST_PROTOCOL)

prediction = cclf.predict(X)
acc = accuracy_score(y, prediction)
print(f'Training Accuracy score: {acc}')

# Read in eval data
eval_data = pd.read_csv(eval_path)
eval_texts = eval_data['tweet']
eval_y = eval_data['ND_label']
eval_x = get_features(eval_data)
proba = cclf.predict_proba(eval_x)
print(proba[1])
proba = np.log(proba)
'''
save_file = eval_path[:-4]+'_ttbias.pkl' 
with open(save_file, 'wb') as handle:
    pickle.dump(proba, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''
prediction = cclf.predict(eval_x)
acc = accuracy_score(eval_y, prediction)
print(f'Eval Accuracy score: {acc}')
