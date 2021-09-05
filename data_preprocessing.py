from sklearn.model_selection import train_test_split
import nltk
from tqdm import tqdm


class SentenceGetter():
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: nltk.pos_tag(s['Tokens'].values.tolist())
        tqdm.pandas()
        self.grouped = self.data.groupby('Sentence ID').progress_apply(agg_func)
        self.tagged_sentences = [s for s in self.grouped]
        agg_func = lambda s: s['Label'].values.tolist()
        tqdm.pandas()
        self.grouped_labels = self.data.groupby('Sentence ID').progress_apply(agg_func)


class CRF_feature():
    def word2features(self, sent, i):
        word = sent[i][0]
        pos_tag = sent[i][1]
        features = {
            'bias': 1.0, 
            'word.lower()': word.lower(), 
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'pos_tag': pos_tag,
            'pos_tag[:2]': pos_tag[:2],
        }
        if i > 0:
            word1 = sent[i-1][0]
            pos_tag1 = sent[i-1][1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:pos_tag': pos_tag1,
                '-1:pos_tag[:2]': pos_tag1[:2],
            })
        else:
            features['BOS'] = True
        if i < len(sent)-1:
            word1 = sent[i+1][0]
            pos_tag1 = sent[i+1][1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:pos_tag': pos_tag1,
                '+1:pos_tag[:2]': pos_tag1[:2],
            })
        else:
            features['EOS'] = True
        return features

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]
