import os
import sh
from io import StringIO
import re
from string import punctuation
import warnings


#warnings.filterwarnings("ignore", category=DeprecationWarning, append=True)
#warnings.filterwarnings("ignore", category=FutureWarning, append=True)
#warnings.filterwarnings("ignore", category=UndefinedMetricWarning, append=True)
warnings.filterwarnings("ignore", category=Warning, append=True)

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.decomposition import (NMF, TruncatedSVD, dict_learning)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pandas as pd

from pdb import set_trace as st
# Hold xml tag syntax items
punctuation = re.sub('[/|<|>]+', '', punctuation)


def build_search_tags(tag_file, return_regexs=True):
    tag_list = []
    with open(tag_dict_f) as f:
        for line in f:
            if "name=" in line:  
                tag_list.append(re.search(name_regex, line).group(1))
            else:
                continue

    if not return_regexs:
        return {tag: ("<{}>".format(tag), "<{}/>".format(tag), id) 
                                    for id, tag in enumerate(tag_list)}
    else:
        return {tag: ("<{}>(.*)<{}/>".format(tag, tag), id) 
                                    for id, tag in enumerate(tag_list)}


def build_window_dataset(lines, tagsregexs, winsize=5, lists=True):

    samples = []
    for line in lines:
        clean = re.sub('['+punctuation+']', '', line)
        clean = re.sub('><', '> <', clean)
        old_list = [i for i in re.findall(r">([^<>]*)</", clean) if i != ' ']
        new_list = [re.sub(' ', '_', item) for item in old_list]
        words = clean
        for old, new in zip(old_list, new_list):
            words = re.sub(old, new, words)

        words = words.split()[1:]
        for idx, word in enumerate(words):
            if word.startswith("<") and ("</" in word):
                tag = re.search("</(.*)>", word).group(1)
                start = idx - min([idx, winsize])
                end = idx + min([idx + len(words) - 1, winsize]) + 1
                sample = ' '.join(words[start:idx] + words[idx + 1:end])
                sample = re.sub('</?(.*)>', '', sample)
                #sample = re.sub('<(.*)>', '', sample)
                if sample != '':
                     samples.append({"text": sample, "label": tag})
    if lists:
        x = []; y = []
        for d in samples:
             x.append(d['text'])
             y.append(d['label'])
        return x, y
    else:
        return samples


def parse_xml(inxml, name_regex = """<xs:element name=\"(.*)\">""",
                     sample_str = "!Sample_growth_protocol_ch1"):

    sample_regex = "/^" + sample_str + "/p"
    if os.path.isdir(inxml):  
        from itertools import filterfalse as filt
        lines = []
        for file in os.listdir(inxml):
            if file.endswith(".xml"):
            #lines = sh.sed("-n", sample_regex, os.path.join(inxml, file))
                buf = StringIO()
                sh.sed("-n", sample_regex, os.path.join(inxml, file), _out=buf)            
                lines.append(buf.getvalue())

        lines = re.sub("\r", '', ' '.join(lines)).split("\n")
    #lines = filt(None, " ".join(lines).split("\n"))

    elif os.path.isfile(inxml):  
        buf = StringIO()
        sh.sed("-n", sample_regex, inxml, _out=buf)
        lines = re.sub("\r", '', buf.getvalue()).split("\n")

    else:
        print("Path does not exist: %s \nEXIT...\n" % inxml)

    while True: # Remove empty lines
        try:
            lines.remove('')
        except ValueError:
            break

    return lines

# Input files
#inxml = "/home/iarroyof/Dropbox/ES_Carlos_Ignacio/xhGCs/paquete-Nacho/ejemplos-etiquetado-xml/GSE54899_family.xml"
#inxml = "/home/iarroyof/Dropbox/ES_Carlos_Ignacio/xhGCs/paquete-Nacho/ejemplos-etiquetado-xml/"
inxml = "/home/iarroyof/data/expConditions/"
#tag_dict_f = "/home/iarroyof/Dropbox/ES_Carlos_Ignacio/xhGCs/paquete-Nacho/ejemplos-etiquetado-xml/esquema-gcs.xsd"
tag_dict_f = "/home/iarroyof/data/expConditions/esquema-gcs.xsd"
stdout = False
# Output results
results = "results.csv"
stats = "stats.csv"
estimator = "estimators.csv"
# Input params
grid = True

name_regex = """<xs:element name=\"(.*)\">"""
sample_str = "!Sample_growth_protocol_ch1"

# Get the tag dictionary from the XSD schema
tag_regexps = build_search_tags(tag_dict_f)

# Get lines containing tagged text.
lines = parse_xml(inxml, name_regex=name_regex, sample_str=sample_str)

# Building dictionary of text samples to be classified. Untagged samples are simply ignored
X_test, y_test = build_window_dataset(lines=lines, tagsregexs=tag_regexps, winsize=5)

# Split data into train and test
if grid:
    X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, 
                                                     test_size=0.33, random_state=42)

# Create a set of classifiers and their available parameters for model selection
    text_clfs = [
        Pipeline([('tfidf', TfidfVectorizer(binary=False, analyzer='char', ngram_range=(1, 4), 
										lowercase=True)),
		  ('gauss', RBFSampler(random_state=1)),
                  ('clf_PA', PassiveAggressiveClassifier())
                ]),
#        Pipeline([('tfidf', TfidfVectorizer(binary=True, analyzer='char', ngram_range=(1, 5),
#                                                                                lowercase=True)),
#                  ('clf_PA', PassiveAggressiveClassifier())
#                ]),
     
        Pipeline([('tfidf', TfidfVectorizer(binary=False, analyzer='char', ngram_range=(1, 4), 
										lowercase=True)), 
                  ('deco_SVD', TruncatedSVD()),
                  ('clf_PA', PassiveAggressiveClassifier())
                ]),
 #       Pipeline([('tfidf', TfidfVectorizer(binary=True, analyzer='char', ngram_range=(1, 5),
 #                                                                               lowercase=True)),
 #                 ('clf_SVM', SVC())
 #               ]),
        Pipeline([('tfidf', TfidfVectorizer(binary=False, analyzer='char', ngram_range=(1, 4),
                                                                                lowercase=True)),
                  ('deco_SVD', TruncatedSVD()),
                  ('clf_SVM', SVC())
                ]),
            ]

    parameters = [
             {
                #'tfidf__ngram_range': [(1, 1), (1, 2), (1,5), (1,4), (2, 4), (2, 5)],
                #'tfidf__sublinear_tf': (True, False),
               #'tfidf__stop_words': (None, 'english'),
               #'tfidf__analyzer': ('word', 'char'),
                #'tfidf__binary': (True, False),
	        'gauss__gamma': (10.0, 1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001),
                'clf_PA__C': (10.0, 50.0, 100.0, 200.0, 300.0, 500.0, 1000.0)

            },
#              {
#                'tfidf__ngram_range': [(1, 1), (1, 2), (1,5), (1,4), (2, 4), (2, 5)],
#                'tfidf__sublinear_tf': (True, False),
                #'tfidf__stop_words': (None, 'english'),
#                'tfidf__analyzer': ('word', 'char'),
#                'tfidf__binary': (True, False),
#                'clf_PA__C': (0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 200.0, 300.0, 500.0)
#
#             },
            {
                #'tfidf__ngram_range': [(1, 1), (1, 2), (1,5), (1,4), (2, 4), (2, 5)],
                #'tfidf__sublinear_tf': (True, False),
                #'tfidf__stop_words': (None, 'english'),
                #'tfidf__analyzer': ('word', 'char'),
                #'tfidf__binary': (True, False),
                'deco_SVD__n_components': (100, 200, 300),
                'clf_PA__C': (10.0, 50.0, 100.0, 200.0, 300.0, 500.0, 1000.0)
             },
 #           {
 #               'tfidf__ngram_range': [(1, 1), (1, 2), (1,5), (1,4), (2, 4), (2, 5)],
 #               'tfidf__sublinear_tf': (True, False),
                #'tfidf__stop_words': (None, 'english'),
 #               'tfidf__analyzer': ('word', 'char'),
 #               'tfidf__binary': (True, False),
 #               'clf_SVM__C': (0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 200.0, 300.0, 500.0, 1000.0)

#             },
            {
                #'tfidf__ngram_range': [(1, 1), (1, 2), (1,5), (1,4), (2, 4), (2, 5)],
                #'tfidf__sublinear_tf': (True, False),
                #'tfidf__stop_words': (None, 'english'),
                #'tfidf__analyzer': ('word', 'char'),
                #'tfidf__binary': (True, False),
                'deco_SVD__n_components': (100, 200, 300),
                'clf_SVM__C': (10.0, 50.0, 100.0, 200.0, 300.0, 500.0, 1000.0)
             },
        ]

    performances=[]
    for text_clf, param in zip(text_clfs, parameters):
        gs_clf = GridSearchCV(text_clf, param, n_jobs=-1, scoring='f1_macro', iid=False, error_score=0.0, verbose=0)
        gs_clf = gs_clf.fit(X_train, y_train)
        predicted = gs_clf.predict(X_test)
        performances.append((gs_clf, predicted, f1_score(y_test, predicted, average='macro')))
        # Add here the feature names and weights for discussion
    clf = sorted(performances, key=lambda x: x[2], reverse=True)[0]
    gs_clf = clf[0]
    predicted = clf[1]
# Save performance statistics and best estimator params
    pd.DataFrame(list(gs_clf.cv_results_.items())).to_csv(stats)
    pd.DataFrame(list(gs_clf.best_params_.items())).to_csv(estimator)

else:
    predicted = gs_clf.predict(X_test)

# imprimir evaluacion de predicciones con el conjunto de test
print("F1_macro: %f\n" % clf[2] if grid else gs_clf)
if stdout:
    print(classification_report(y_test, predicted))
else:
    #pd.DataFrame(list(classification_report(y_test, predicted, 
    #                            output_dict=True).items())).to_csv(results)
    import csv
    d = classification_report(y_test, predicted, output_dict=True)
    headers = ['class'] + [list(d[header].keys()) for header in d][0]
    R = [headers] + [[D] + list(d[D].values()) for h in headers for D in d]
    with open(results, "w") as fo:
        writer = csv.writer(fo)
        writer.writerows(R)
