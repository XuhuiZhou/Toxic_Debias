import pandas as pd
import numpy as np
import csv
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import sys

if len(sys.argv)>1:
    training_path = sys.argv[1]
else:
    training_path = 'toxic_language/advToxicityFilt_v0.2/ND_davidson_trn_dial_new.csv' 

print(training_path.split('/')[-2])
eval_path = 'toxic_language/advToxicityFilt_v0.2/ND_davidson_tst_dial_new.csv'
saved_path='bow_model'

#toxic_identity_list = ['blind','young','asian','male','muslim','latino','female','gay','queer','islam' ,'lesbian' ,'latina'] 
dirty_words_path = 'toxic_language/advToxicityFilt_v0.2/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/en'
dirty_words = open(dirty_words_path, 'r').read().splitlines()

identity_words_path = 'toxic_language/advToxicityFilt_v0.2/unintended_bias/adjectives_people.txt'
identity_words = open(identity_words_path, 'r').read().splitlines()

def get_feature(texts, dialect):
    words = texts.lower().split()
    feature = []
    feature.append(dialect)
    ''' 
    for i in identity_words:
        if i in words:
             feature.append(i)
    
    for i in words:
        if i in dirty_words:
            feature.append(i)
    ''' 
    return ' '.join(feature)
    
def read_csv(data):
    rep = []
    labels = []
    texts = []
    features = []
    dialect_tag = []
    with open(data, 'r') as csvfile:
        reader=csv.reader(csvfile)
        next(reader)
        for row in reader:
            label = int(row[-1].strip()) 
            labels.append(label)
            texts.append(row[-2])
            dialect = row[-3]
            feature = get_feature(row[-2], dialect)
            features.append(feature)  
    #labels = labels[:400]
    #print(labels)
    print(len(labels))
    toxic = sum(labels)
    toxic_ratio = toxic/len(labels)
    ratio = [1-toxic_ratio, toxic_ratio]
    print(ratio)
    labels = np.array(labels)
    #print(labels)
    return {'tweet':texts, 'ND_label':labels, 'features':features} 


# Read in data
data = read_csv(training_path)
texts = data['tweet']
y = data['ND_label']
F = data['features']

# Vectorize the text
vectorizer = CountVectorizer(stop_words='english', min_df=0.0001)
X = vectorizer.fit_transform(F)

# Train the model
model = LinearSVC(class_weight="balanced", dual=False, tol=1e-2, max_iter=1e5)
cclf = CalibratedClassifierCV(base_estimator=model)
cclf.fit(X, y)

# Save the model
#joblib.dump(vectorizer, saved_path+'/'+'vectorizer.joblib')
#joblib.dump(cclf, saved_path+'/'+'model.joblib')


#vectorizer = joblib.load(pkg_resources.resource_filename('profanity_check', 'data/vectorizer.joblib'))
#model = joblib.load(pkg_resources.resource_filename('profanity_check', 'data/model.joblib'))

def _get_profane_prob(prob):
  return prob[1]

def predict(model, vectorizer,  texts):
  return model.predict(vectorizer.transform(texts))

def predict_prob(model, vectorizer, texts):
  return np.apply_along_axis(_get_profane_prob, 1, model.predict_proba(vectorizer.transform(texts)))

# Record the training bias
proba = cclf.predict_proba(X)
print(proba[1])
proba = np.log(proba)
save_file = training_path[:-4]+'_dpfbias.pkl' 
with open(save_file, 'wb') as handle:
    pickle.dump(proba, handle, protocol=pickle.HIGHEST_PROTOCOL)
prediction = predict(cclf,vectorizer, F)
acc = accuracy_score(y, prediction)
print(f'Training Accuracy score: {acc}')

# Read in eval data
eval_data = read_csv(eval_path)
eval_texts = eval_data['tweet']
eval_y = eval_data['ND_label']
eval_x = eval_data['features']
prediction = predict(cclf,vectorizer, eval_x)
acc = accuracy_score(eval_y, prediction)
print(f'Eval Accuracy score: {acc}')


