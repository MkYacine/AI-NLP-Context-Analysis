import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_20newsgroups
##############################################
# PARTIE 1
##############################################

#Prep for main project, not related to problem
def preparatifs():
    dataset = load_iris()
    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # SVM
    from sklearn import svm

    svm = svm.SVC()
    svm.fit(X_train, y_train)  # Entrainement du modele SVM
    print("SVM results:")
    svm.score(X_test, y_test)  # Predictions du SVM

    # Reseau Neuronal
    # https://stackoverflow.com/questions/46028914/multilayer-perceptron-convergencewarning-stochastic-optimizer-maximum-iterat
    # Selection de parametres optimales:
    param_grid = [{
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'hidden_layer_sizes': [
            (1,), (2,), (3,), (4,), (5,), (6,), (7,)
        ]}]
    clf = GridSearchCV(MLPClassifier(max_iter=300), param_grid, cv=3,
                       scoring='accuracy')
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print(clf.best_params_)
    # Nous a donné:
    # Best parameters set found on development set:
    # {'activation': 'identity', 'hidden_layer_sizes': (3,), 'solver': 'sgd'}
    network = MLPClassifier(max_iter=2000, hidden_layer_sizes=(3,),
                            activation='identity', solver='sgd')
    network.fit(X_train, y_train)  # Entrainement du reseau neuronal
    print("Neural Network results:")
    print(network.score(X_test, y_test)) # Prediction du reseau neuronal

    # Arbre De Decision
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)  # Entrainement de l'arbre de decision
    print("Decision tree results:")
    tree.score(X_test, y_test)# Prediction de l'arbre de decision

    # Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train, y_train)  # Entrainement du Naive Bayes
    print("Naive Bayes results:")
    print(nb.score(X_test, y_test))  # Prediction du Naive Bayes

    # Random forest
    rfc = RandomForestClassifier(n_estimators=10)
    rfc.fit(X_train, y_train)  # Entrainement de la foret aleatoire
    print("Random Forest results:")
    print(rfc.score(X_test, y_test))  # Prediction du Naive Bayes

    # 1.3. Selection et ponderation des features ----------------------------------

    # https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

    stopWords = Path('./stoplist').read_text().split("\n")
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    textData = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

    count_vect = CountVectorizer(stop_words=stopWords)
    analyze = count_vect.build_analyzer()
    stemmer = PorterStemmer()

    for i in range(len(textData.data)):
        text = textData.data[i]
    arr = analyze(text)
    for j in range(len(arr)):
        arr[j] = stemmer.stem(arr[j])
    textData.data[i] = " ".join(arr)

    X = textData.data
    y = textData.target
    X_counts = count_vect.fit_transform(X)

    tf_transformer_train = TfidfTransformer(use_idf=False).fit(X_counts)

    X_tf = tf_transformer_train.transform(X_counts)
    X_train, X_test, y_train, y_test = train_test_split(X_tf, y, test_size=0.3, random_state=42)
    textModel = MultinomialNB()
    textModel.fit(X_train, y_train)
    print(textModel.score(X_test, y_test))

##############################################
# PARTIE 2
##############################################
#Traitement des donnees avec un dictionnaire et dataframe
def formatDict(wordFilter, stemming, before, after):
    txt = Path('./interest').read_text()
    if wordFilter:
        stopWords = Path('./stoplist').read_text().split('\n')
    else:
        stopWords = []

    txt = txt.replace("======================================", "")
    txt = txt.replace("[ ", "")
    txt = txt.replace("] ", "")
    txt = txt.replace("\n", "")
    dataset = txt.split("$$")
    formattedData = []
    stemmer = PorterStemmer()

    # Formattage du texte
    # Nous donnera une liste de forme [ #Phrase1[(Mot1,Tag1), (Mot2,Tag2), (Mot3, Tag3)], ... ]
    for data in dataset:
        # Separation des mots
        curr = data.split(" ")
        currData = []
        for ele in curr:
            # Separation de chaque element de la phrase en paire mot/forme gramaticale
            pair = ele.split("/")
            if len(pair) != 2:
                continue
            # Filtrage des stops words et ponctuation
            if pair[0] in "!#$&'()*+, -./:;<=>?@[\]^_`{|}~":
                continue
            if pair[0] in stopWords:
                continue
            if stemming:
                pair[0] = stemmer.stem(pair[0])
            # Formattage des interest
            if ("interest" in pair[0]):
                pair[0] = pair[0].replace("s_", "")
                pair[0] = pair[0].replace("_", "")
            # Remplacement des punct par 'X'
            for punct in "!#$&'()*+, -./:;<=>?@[\]^_`{|}~":
                if punct in pair[0]:
                    pair[0] = pair[0].replace(punct, "X")
            if "%" in pair[0]:
                pair[0] = pair[0].replace("%", "PERCENT")
            currData.append(pair)
        formattedData.append(currData)
    # Preparation des colonnes du dataframe
    columns = []
    for i in range(1, before + 1):
        columns.append("word(" + str(-i) + ")")
        columns.append("tag(" + str(-i) + ")")
    for i in range(1, after + 1):
        columns.append("word(" + str(i) + ")")
        columns.append("tag(" + str(i) + ")")
    columns.append("target")

    # Obtention des mention de interest ainsi que les mots entourant
    # Nous donnera une liste de forme [ #Mention1DeInterest [ MotAvant1, TagAvant1, MotApres1, TagApres1, Contexte], ...]
    entries = []
    for data in formattedData:
        for i in range(len(data)):
            if ("interest" in data[i][0]) and (data[i][0][-1] in "123456"):
                entry = []
                for j in range(1, before + 1):
                    try:
                        entry.append(data[i - j][0])
                        entry.append(data[i - j][1])
                    except:
                        # Quand il n y a plus de mot, en mets None
                        # Ceci nous peremettera de prendre en compte la position du mot interest dans la phrase
                        # (Vers le debut ou vers la fin)
                        entry.append(None)
                        entry.append(None)
                for j in range(1, after + 1):
                    try:
                        entry.append(data[i + j][0])
                        entry.append(data[i + j][1])
                    except:
                        entry.append(None)
                        entry.append(None)
                entry.append(data[i][0][-1])
                entries.append(entry)

    # Preparation d'un dictionnaire du vocabulaire
    # Pour transformer les donnees de chaines en entiers
    wordSet = set()
    for entry in entries:
        wordSet.update(entry)
    # https://stackoverflow.com/questions/28016752/sklearn-trying-to-convert-string-list-to-floats
    dictionary = dict(zip(wordSet, range(len(wordSet))))  # assign each string an integer, and put it in a dict

    # Les X formattes en forme [ #Entry1 [ReferenceMot1, ReferenceMot2, ReferenceMot3...] ...]
    formattedEntries = [[dictionary[word] for word in entry[:((before + after) * 2)]] for entry in
                        entries]  # store class labels as ints
    # Les Y (Contexte du mots interest
    targets = [entry[-1] for entry in entries]
    df = pd.DataFrame(columns=columns)
    for i in range(len(entries)):
        df.loc[i] = formattedEntries[i] + [targets[i]]
    return df

#Traitement des donnees avec CountVectorizer et TfidVectorizer
def formatCountVect(wordFilter, stemming, before, after):
    txt = Path('./interest').read_text()
    if wordFilter:
        stopWords = Path('./stoplist').read_text().split('\n')
    else:
        stopWords = []
    txt = txt.replace("======================================", "")
    txt = txt.replace("[ ", "")
    txt = txt.replace("] ", "")
    txt = txt.replace("\n", "")
    dataset = txt.split("$$")
    stemmer = PorterStemmer()
    count_vect = CountVectorizer()
    formattedData=[]
    for data in dataset:
        # Separation des mots
        curr = data.split(" ")
        currData = []
        for ele in curr:
            # Separation de chaque element de la phrase en paire mot/forme gramaticale
            pair = ele.split("/")
            if len(pair) != 2:
                continue
            # Filtrage des stops words et ponctuation
            if pair[0] in "!#$&'()*+, -./:;<=>?@[\]^_`{|}~":
                continue
            if pair[0] in stopWords:
                continue
            if stemming:
                pair[0] = stemmer.stem(pair[0])
            # Formattage des interest
            if ("interest" in pair[0]):
                pair[0] = pair[0].replace("s_", "")
                pair[0] = pair[0].replace("_", "")
            # Remplacement des punct par 'X'
            for punct in "!#$&'()*+, -./:;<=>?@[\]^_`{|}~":
                if punct in pair[0]:
                    pair[0] = pair[0].replace(punct, "X")
            if "%" in pair[0]:
                pair[0] = pair[0].replace("%", "PERCENT")
            currData.append(pair)
        formattedData.append(currData)
    # Obtention des mention de interest ainsi que les mots entourant
    # Nous donnera une liste de forme [ #Mention1 "MotAvant1 TagAvant1 MotApres1 TagApres1", #Mention2 ...]
    entries = []
    for data in formattedData:
        for i in range(len(data)):
            if ("interest" in data[i][0]) and (data[i][0][-1] in "123456"):
                entry = ""
                for j in range(1, before + 1):
                    try:
                        entry+=(data[i-j][0])+" "+(data[i - j][1])+" "
                    except:
                        pass
                for j in range(1, after + 1):
                    try:
                        entry+=(data[i+j][0])+" "+(data[i+j][1])+" "
                    except:
                        pass
                entries.append(entry)
    X_counts = count_vect.fit_transform(entries)
    tf_transformer_train = TfidfTransformer(use_idf=False).fit(X_counts)

    X_tf = tf_transformer_train.transform(X_counts)
    return X_tf


#Entraine un model et imprime son score
def trainScoreModel (model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))

#Trouve les hyperparametres optimales pour un modele
def hypertuneModel(model, param_grid, X_train, y_train):
    gridsearch = GridSearchCV(model, param_grid, cv=5,
                           scoring='accuracy')
    gridsearch.fit(X_train, y_train)
    print(gridsearch.best_score_)
    return(gridsearch.best_params_)



##########################################################
# Main function
##########################################################

#0) Preparatifs
#Les prepartifs effectues avant le probleme principal
#preparatifs()

#1) Pretraitement
df = formatDict(wordFilter=False, stemming=False, before=4, after=4)
X = df.loc[:, df.columns != 'target']
X_tf = formatCountVect(wordFilter=False, stemming=False, before=4, after=4)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_tf_train, X_tf_test, y_tf_train, y_tf_test = train_test_split(X_tf, y, test_size=0.2, random_state=42)

#Hyperparametres a tester:
rfc_param_grid = [
        {
            'n_estimators':[
              40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240
             ]}]
mlp_param_grid = [
        {
            'activation' : ['logistic','tanh','relu'],
            'solver' : ['adam'],
            'hidden_layer_sizes': [(10,10,10), (100,50,10), (20,), (10,10,10,10), (10,10,10,10,10),(10,10)]}]

#2) Comparaisons des deux dataset
tree = DecisionTreeClassifier()
svm = svm.SVC()
nb = MultinomialNB()
mlp = MLPClassifier()
rfc = RandomForestClassifier()
models = {"Tree":tree,
          "SVM":svm,
          "Naive Bayes":nb,
          "Reseau de neurones": mlp,
          "Foret Aleatoire": rfc}
for key, value in models.items():
    print("Score du "+key+" avec le dataset Dict: ")
    trainScoreModel(value, X, y)
    print("Score du "+key+" avec le dataset CountVect: ")
    trainScoreModel(value, X_tf, y)



#3) HyperTuning
rfc_best_params= hypertuneModel(RandomForestClassifier(), rfc_param_grid, X_tf_train, y_tf_train)
print(rfc_best_params)
mlp_best_params= hypertuneModel(MLPClassifier(max_iter=300, random_state=1), mlp_param_grid, X_tf_train, y_tf_train)
print(mlp_best_params)