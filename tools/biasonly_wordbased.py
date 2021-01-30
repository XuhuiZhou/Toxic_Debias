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

mode = sys.argv[1]

training_path = sys.argv[2]
word_list = sys.argv[4]

#offensive-minority-reference offensive-not-minority harmless-minority  
word_type = sys.argv[5]

df = pd.read_csv(word_list)
df = df[df.categorization==word_type]
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

# Record the training bias

proba = cclf.predict_proba(X)
proba = np.log(proba)
save_file = training_path[:-4]+'_kkk.pkl' 
with open(save_file, 'wb') as handle:
    pickle.dump(proba, handle, protocol=pickle.HIGHEST_PROTOCOL)

prediction = cclf.predict(X)
acc = accuracy_score(y, prediction)
print(f'Training Accuracy score: {acc}')

