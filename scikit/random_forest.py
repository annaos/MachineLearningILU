import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = '../data/'
TRAINSET_PATH = DATA_PATH + 'train_set_original.csv'

data = pd.read_csv(TRAINSET_PATH)

feature_list = ['rows', 'nonzeros', 'posdef', 'nsym', 'psym', 'density',
                 'avg_nnz', 'max_nnz', 'std_nnz',
                 'avg_row_block_count', 'std_row_block_count', 'min_row_block_count',
                 'max_row_block_count',
                 'avg_row_block_size', 'std_row_block_size', 'min_row_block_size',
                 'max_row_block_size',
                 'block_count']

features = np.array(data[feature_list])
labels = np.array(data.isEffective)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2)#, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

rf = RandomForestClassifier(n_estimators = 1000)#, random_state = 42)
rf.fit(train_features, train_labels)


# plot feature importance
plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.25)
feature_imp = pd.Series(rf.feature_importances_,index=feature_list).sort_values(ascending=False)
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()

predictions = rf.predict(test_features)

# Calculate errors
prediction_effective = predictions.sum().item()
label_effective = test_labels.sum().item()

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
print('Confusion matrix:')
print(np.matrix(confusion_matrix))

print('Error, where predicted effective (false positive): ', false_positives)
print('Error, where label effective (false negative): ', false_negatives)
print('Accuracy: ', accuracy, ' (everytime 0 has accurancy:', 100 - 100*label_effective/len(test_labels), ')' )
print('Precision: ', precision)
print('Recall: ', recall)
print('F_one: ', f_one)
