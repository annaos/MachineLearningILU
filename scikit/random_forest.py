import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.tree import export_graphviz

from dataset.utils import Utils

DATA_PATH = '../data/'
TRAINSET_PATH = DATA_PATH + 'train_set.csv'
FEATURE_COLLECTIONS = ['relative', 'normalized', 'relative_normalized', 'all', 'original']


def get_original_dataset(feature_list, type='original'):
    if type == 'balanced':
        original_data = pd.read_csv(DATA_PATH + 'train_set_original_balanced.csv')
    elif type == 'balanced_new':
        original_data = pd.read_csv(DATA_PATH + 'train_set_original_balanced_new_normalized.csv')
    else:
        original_data = pd.read_csv(DATA_PATH + 'train_set_original.csv')
    original_features = np.array(original_data[feature_list])
    original_labels = np.array(original_data.isEffective)
    return original_features, original_labels, original_data.ProblemId


def get_train_dataset(feature_list):
    data = pd.read_csv(TRAINSET_PATH)

    features = np.array(data[feature_list])
    labels = np.array(data.isEffective)

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2) # , random_state = 42)

    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)
    return train_features, test_features, train_labels, test_labels


def plot_feature_importances(feature_importances, feature_list, name):
    # plot feature importance
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.25)
    feature_imp = pd.Series(feature_importances, index=feature_list).sort_values(ascending=False)
    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.savefig(f'../data/feature_importance_2_{name}.png')
    plt.show()


def evaluation(test_labels, predictions):
    # Calculate errors
    prediction_effective = predictions.sum().item()
    label_effective = test_labels.sum().item()

    with np.errstate(divide='ignore', invalid='ignore'):
        confusion_vector = predictions / test_labels
    true_positives = sum(confusion_vector == 1).item()
    false_positives = sum(confusion_vector == float('inf')).item()
    true_negatives = sum(np.isnan(confusion_vector)).item()
    false_negatives = sum(confusion_vector == 0).item()
    confusion_matrix = [[true_positives, false_negatives], [false_positives, true_negatives]]

    accuracy = 100 * (true_positives + true_negatives) / len(test_labels)
    if true_positives != 0:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f_one = 2 * precision * recall / (precision + recall)
    else:
        precision, recall, f_one = 0, 0, 0

    print('Test set: ', len(test_labels))
    print('Label effective: ', label_effective)
    print('Predicted effective: ', prediction_effective)
    print('Errors: ', false_negatives + false_positives)
    print(f'Confusion matrix: $\\begin{{array}}{{rrr}} {true_positives} & {false_negatives} \\\\ {false_positives} & {true_negatives} \\\\ \\end{{array}}$')
    print(np.matrix(confusion_matrix))

    print('Error, where predicted effective (false positive): ', false_positives)
    print('Error, where label effective (false negative): ', false_negatives)
    print('Accuracy: ', accuracy, ' (everytime 0 has accuracy:', 100 - 100 * label_effective / len(test_labels), ')')
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F_one: ', f_one)


def print_false(test_labels, predictions, data):
    f_p = []
    f_n = []
    t_p = []
    t_n = []
    for i, prediction in enumerate(predictions):
        if test_labels[i] == 0 and predictions[i] == 1:
            f_p.append(data[i])
        if test_labels[i] == 1 and predictions[i] == 0:
            f_n.append(data[i])
        if test_labels[i] == 1 and predictions[i] == 1:
            t_p.append(data[i])
        if test_labels[i] == 0 and predictions[i] == 0:
            t_n.append(data[i])
    print(f'false positive ({len(f_p)}): {f_p}')
    print(f'false negative ({len(f_n)}): {f_n}')
    print(f'true positive ({len(t_p)}): {t_p}')
    print(f'true negative ({len(t_n)}): {t_n}')


def train(classifier, feature_art):
    feature_list = Utils.get_features_list(feature_art)
    train_features, test_features, train_labels, test_labels = get_train_dataset(feature_list)

    classifier.fit(train_features, train_labels)
    if hasattr(classifier, 'feature_importances_'):
        plot_feature_importances(classifier.feature_importances_, feature_list, feature_art)

    predictions = classifier.predict(test_features)
    evaluation(test_labels, predictions)

    print('-----ORIGINAL BALANCED NEW DATASET----------')
    original_balanced_features, original_balanced_labels, problem_ids = get_original_dataset(feature_list, 'balanced_new')
    predictions = classifier.predict(original_balanced_features)
    evaluation(original_balanced_labels, predictions)
    print_false(original_balanced_labels, predictions, problem_ids)

    print('-----ORIGINAL BALANCED DATASET----------')
    original_balanced_features, original_balanced_labels, problem_ids = get_original_dataset(feature_list, 'balanced')
    predictions = classifier.predict(original_balanced_features)
    evaluation(original_balanced_labels, predictions)
    print_false(original_balanced_labels, predictions, problem_ids)

    print('-----ORIGINAL DATASET----------')
    original_features, original_labels, problem_ids = get_original_dataset(feature_list)
    predictions = classifier.predict(original_features)
    evaluation(original_labels, predictions)
    print_false(original_labels, predictions, problem_ids)


clf = RandomForestClassifier(n_estimators = 1000)

#clf = GradientBoostingClassifier(
#    n_estimators=1000,
    # max_depth=
    # max_leaf_nodes=
    # min_samples_leaf=
    # min_samples_split=
    #n_iter_no_change=50,
    #init = 'zero',
    #max_features=0.5 # 0,5 better than 0,8 or sqrt
#)

# clf = svm.SVC()
for feature_art in FEATURE_COLLECTIONS:
    print('------------------FEATURE ART------------------:', feature_art)
    train(clf, feature_art)
