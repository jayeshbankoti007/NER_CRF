import numpy as np
import pandas as pd
from sklearn_crfsuite import metrics

def load_data(file_path):
    ner_data = pd.read_csv(file_path, encoding = "ISO-8859-1")
    return ner_data


def clean_data(ner_data):
    ner_data['Word'] = ner_data['Word'].str.strip()
    ner_data['Word'].replace('', np.nan, inplace=True)
    ner_data.isnull().sum()
    ner_data = ner_data.dropna(axis=0, how='any')
    return ner_data

def visualize_data(ner_data):
    #For visualizing the data
    print("Data Head")
    print(ner_data.head())
    print("Sentence IDs")
    print(ner_data['Sentence ID'].nunique())
    print("Null data: ")
    print(ner_data.isnull().sum())
    print("Tokens or Word Count")
    print(ner_data['Tokens'].nunique() )
    print("Uniqua Labels")
    print(ner_data.Label.nunique())
    print("Label Info")
    print(ner_data.groupby('Label').size().reset_index(name='counts'))


def get_accuracy_score(input, expected, model):
    y_pred = model.predict(input)
    model_output = metrics.flat_classification_report(expected, y_pred, output_dict=True)
    return model_output
