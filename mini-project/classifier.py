import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from data_cleaning import load_data

classifier_list = ['MNB', 'GNB', 'KNN', 'SVM', 'DT', 'ET', 'GB', 'RF', 'BC', 'LR', 'RC']
cross_validation = 3
num_jobs = 2
random_state = 0

def classify(type):
    train_vectors, test_vectors, train_labels, test_labels = load_data()

    classifier = None
    if (type == 'MNB'):
        classifier = MultinomialNB(alpha=0.1)
        classifier.fit(train_vectors, train_labels)
    elif (type == 'GNB'):
        classifier = GaussianNB()
        classifier.fit(train_vectors, train_labels)
    elif (type == 'KNN'):
        classifier = KNeighborsClassifier()
        params = {'n_neighbors': [1, 2, 3], 'weights': ['uniform', 'distance']}
        classifier = GridSearchCV(classifier, params, cv=cross_validation, n_jobs=num_jobs)
        classifier.fit(train_vectors, train_labels)
        classifier = classifier.best_estimator_
    elif (type == 'SVM'):
        classifier = SVC()
        params = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10]}
        classifier = GridSearchCV(classifier, params, cv=cross_validation, n_jobs=num_jobs)
        classifier.fit(train_vectors, train_labels)
        classifier = classifier.best_estimator_
    elif (type == 'DT'):
        classifier = DecisionTreeClassifier(max_depth=20, min_samples_split=2, random_state=random_state)
        params = {'criterion': ['gini', 'entropy']}
        classifier = GridSearchCV(classifier, params, cv=cross_validation, n_jobs=num_jobs)
        classifier.fit(train_vectors, train_labels)
        classifier = classifier.best_estimator_
    elif (type == 'ET'):
        classifier = ExtraTreesClassifier(max_depth=20, min_samples_split=2, random_state=random_state)
        params = {'criterion': ['gini', 'entropy']}
        classifier = GridSearchCV(classifier, params, cv=cross_validation, n_jobs=num_jobs)
        classifier.fit(train_vectors, train_labels)
        classifier = classifier.best_estimator_
    elif (type == 'GB'):
        classifier = GradientBoostingClassifier(max_depth=20, min_samples_split=2, random_state=random_state)
        params = {'criterion': ['friedman_mse', 'mse', 'mae']}
        classifier = GridSearchCV(classifier, params, cv=cross_validation, n_jobs=num_jobs)
        classifier.fit(train_vectors, train_labels)
        classifier = classifier.best_estimator_
    elif (type == 'RF'):
        classifier = RandomForestClassifier(max_depth=20, min_samples_split=2, random_state=random_state)
        params = {'n_estimators': [n for n in range(5, 50, 5)], 'criterion': ['gini', 'entropy']}
        classifier = GridSearchCV(classifier, params, cv=cross_validation, n_jobs=num_jobs)
        classifier.fit(train_vectors, train_labels)
        classifier = classifier.best_estimator_
    elif (type == 'BC'):
        classifier = BaggingClassifier(random_state=random_state)
        params = {'n_estimators': [n for n in range(5, 50, 5)]}
        classifier = GridSearchCV(classifier, params, cv=cross_validation, n_jobs=num_jobs)
        classifier.fit(train_vectors, train_labels)
        classifier = classifier.best_estimator_
    elif (type == 'LR'):
        classifier = LogisticRegression(multi_class='auto', solver='newton-cg', random_state=random_state)
        params = {"C": np.logspace(-5, 3, 20, base=2), "penalty": ["l2"]}
        classifier = GridSearchCV(classifier, params, cv=cross_validation, n_jobs=num_jobs)
        classifier.fit(train_vectors, train_labels)
        classifier = classifier.best_estimator_
    elif (type == 'RC'):
        classifier = RidgeClassifier(alpha=1.0, solver='auto', random_state=random_state)
        classifier.fit(train_vectors, train_labels)
    else:
        print("Classifier Type Not Included In This Project!")
        return

    accuracy = accuracy_score(train_labels, classifier.predict(train_vectors))
    print("Training Accuracy:", accuracy)
    test_predictions = classifier.predict(test_vectors)
    accuracy = accuracy_score(test_labels, test_predictions)
    print("Test Accuracy:", accuracy)
    print("Confusion Matrix:", )
    print(confusion_matrix(test_labels, test_predictions))

def test_all():
    for i in range(len(classifier_list)):
        print('\n')
        print("Classifier:", classifier_list[i])
        classify(classifier_list[i])


test_all()