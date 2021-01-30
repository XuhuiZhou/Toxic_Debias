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

mode = sys.argv[1]

training_path = sys.argv[2]

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

# Record the training bias

proba = cclf.predict_proba(X)
proba = np.log(proba)
save_file = training_path[:-4]+'_dddbias.pkl' 
with open(save_file, 'wb') as handle:
    pickle.dump(proba, handle, protocol=pickle.HIGHEST_PROTOCOL)

prediction = cclf.predict(X)
acc = accuracy_score(y, prediction)
print(f'Training Accuracy score: {acc}')

