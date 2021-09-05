import numpy as np
import pandas as pd
import sklearn_crfsuite
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import eli5
from data_utils import clean_data, visualize_data, load_data, get_accuracy_score
from data_preprocessing import SentenceGetter, CRF_feature
import time

train_ner_data = load_data("data/disease_extraction/train.csv")
test_ner_data = load_data("data/disease_extraction/test.csv")

train_ner_data = clean_data(train_ner_data)
test_ner_data = clean_data(test_ner_data)

visualize_data(train_ner_data)
visualize_data(test_ner_data)


#Get your Y labels
y = train_ner_data.tag.values

classes = np.unique(y)
classes = classes.tolist()
new_classes = set(classes.copy())
new_classes.remove('O')
new_classes


#Create feature vectors
getter = SentenceGetter(train_ner_data)
X = [CRF_feature().sent2features(s) for s in tqdm(getter.tagged_sentences)]
y = [s for s in getter.grouped_labels]

getter = SentenceGetter(test_ner_data)
X_test = [CRF_feature().sent2features(s) for s in tqdm(getter.tagged_sentences)]
y_test = [s for s in getter.grouped_labels]

#create test train data
X_train, X_val, X_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

#Train the model

c1 = np.linspace(0, 1, 10)
c2 = np.linspace(0, 1, 10)

for val1 in c1:
    for val2 in c2:
        model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=val1,
            c2=val2,
            max_iterations=100,
            all_possible_transitions=True
        )
        model.fit(X_train, X_train)
        print(get_accuracy_score(X_val, y_val, model))
        time.sleep(1)

#Test Validation score
model_output = get_accuracy_score(X_test, y_test, model)

#Dump Output
df = pd.read_json(json.dumps(model_output))
df.to_csv('output/csv_files/basic_crf_disease_model_output.csv')



expl = eli5.explain_weights(model, top=30)
print(eli5.format_as_text(expl))