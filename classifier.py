# Instructions
# Run classifier.py in a directory containing a folder named "anonymisedData" containing all excel documents from OU
# A folder should be created after execution of the program named "images" and inside this "Machine Learning Pictures" 
# This is where all the figures are saved 

''' Required Packages: 

- sklearn
- matplotlib
- numpy 
- pandas
- seaborn
- graphviz

'''

''' Miscellaneous ''' 

# Python >= 3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
datapath = os.path.join(sys.path[0], "anonymisedData")

# to make this notebook's output stable across runs
np.random.seed(42)

# Where to save the figures
PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", "Machine Learning Pictures")
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


''' Combining Data ''' 

import pandas as pd

# Excluding studentVle as it gives Memory Error + vle as it is relational to studentVle 
# Therefore, we combine: assessments, courses, studentAssessment, studentInfo, studentRegistration 

# Create a "master" df where we will combine all our excel documents 

# Initialise data from excel docs 
assessments = pd.read_csv(datapath + "/" + "assessments.csv") 
courses = pd.read_csv(datapath + "/" + "courses.csv") 
studentAssessment = pd.read_csv(datapath + "/" + "studentAssessment.csv") 
studentInfo = pd.read_csv(datapath + "/" + "studentInfo.csv")
studentRegistration = pd.read_csv(datapath + "/" + "studentRegistration.csv") 

# Remove unneccarily column 
studentRegistration = studentRegistration.drop(columns=['date_unregistration']) 


# Merge all dataframes into a single dataframe named "master" 
master = studentAssessment.join(assessments.set_index('id_assessment'), on='id_assessment') 
master = master.drop(columns=['code_module', 'code_presentation'])
master = master.join(studentInfo.set_index('id_student'), on='id_student') 
merged = studentRegistration.merge(courses, on=['code_module', 'code_presentation'])
merged = merged.drop(columns=['code_module', 'code_presentation'])
master = master.join(merged.set_index('id_student'), on='id_student')

''' Data Formatting ''' 

test = studentInfo
test.head()

from sklearn.preprocessing import LabelEncoder
# Format the data so features are in integer form

# Account for 'nan' data in imd_band 
test['imd_band'].fillna(test['imd_band'].mode().values[0], inplace = True)
test['imd_band'] = test['imd_band'].replace(np.nan, 0)
test['imd_band'] = test['imd_band'].str.replace(r'%', '')

# Converting all data to numerical data using label encoders 

le_imd_band = LabelEncoder()
le_final_result = LabelEncoder()
le_code_module = LabelEncoder()
le_code_presentation = LabelEncoder() 
le_id_student = LabelEncoder()
le_gender = LabelEncoder()
le_region = LabelEncoder()
le_highest_education = LabelEncoder()
le_imd_band = LabelEncoder()
le_age_band = LabelEncoder()
le_disability = LabelEncoder()
le_code_module = LabelEncoder()
le_num_of_prev_attempts = LabelEncoder() 
le_studied_credits = LabelEncoder() 
 
test['code_module_n'] = le_code_module.fit_transform(test['code_module'])
test['code_presentation_n'] = le_code_presentation.fit_transform(test['code_presentation'])
test['id_student_n'] = le_id_student.fit_transform(test['id_student'])
test['gender_n'] = le_gender.fit_transform(test['gender'])
test['region_n'] = le_region.fit_transform(test['region'])
test['highest_education_n'] = le_highest_education.fit_transform(test['highest_education'])
test['imd_band_n'] = le_imd_band.fit_transform(test['imd_band'])
test['age_band_n'] = le_age_band.fit_transform(test['age_band'])
test['num_of_prev_attempts_n'] = le_num_of_prev_attempts.fit_transform(test['num_of_prev_attempts'])
test['studied_credits_n'] = le_studied_credits.fit_transform(test['studied_credits']) 
test['disability_n'] = le_disability.fit_transform(test['disability'])
test['final_result_n'] = le_final_result.fit_transform(test['final_result'])

test = test.drop(['code_module', 'code_presentation', 
                        'id_student', 'imd_band', 'gender', 
                        'region', 'highest_education', 'age_band', 
                        'disability', 'final_result', 'num_of_prev_attempts', 'studied_credits'],axis='columns')


test.head() 

# Training dataset based on Distinction, Fail, Pass, Widthdrawn
data = test.drop('final_result_n', axis='columns')
target = test.final_result_n

X = data.iloc[:,] # This select all features in data
y = target

# Discuss preventing overfitting via generalisation (test_train_split)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision tree 
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train,y_train)
test_score = tree_clf.score(X_test,y_test)
test_score # Containing Distinction, Fail, Pass, Withdrawn


# Format the data so the features are in integer form using label encoders.

'''
Gender:
0 = Female, 1 = Male

Highest Education:
0 = A level or equivalent, 1 = HE Qualification, 2 = Lower than A level,
3 = No Formal quals , 4 = Post Graduate Qualification

Age band:
0-35 = 0, 35-55 =1, 55<= =2

Disability:
0=No, 1=Yes

Region:
0 = East Anglian, 1 = East Midlands, 3 = London,
2 = Ireland, 4 = North, 5 = North West, 6 = Scotland, 
7 = South East, 8 = South, 9 = South West, 10 = Wales, 
11 = West Midlands, 12 = Yorkshire

Final result: 
0 = Distinction 
1 = Fail
2 = Pass
3 = Withdrawn 


'''
# Account for 'nan' data in imd_band 
master['imd_band'].fillna(master['imd_band'].mode().values[0], inplace = True)
master['imd_band'] = master['imd_band'].replace(np.nan, 0)
master['imd_band'] = master['imd_band'].str.replace(r'%', '')

# Converting all data to numerical data using label encoders 

le_assessment_type = LabelEncoder() 
le_code_module = LabelEncoder() 
le_code_presentation = LabelEncoder() 
le_gender = LabelEncoder() 
le_region = LabelEncoder() 
le_highest_education = LabelEncoder() 
le_imd_band = LabelEncoder() 
le_age_band = LabelEncoder() 
le_disability = LabelEncoder() 
le_final_result = LabelEncoder() 


master['assessment_type_n'] = le_assessment_type.fit_transform(master['assessment_type'])
master['code_module_n'] = le_code_module.fit_transform(master['code_module'])
master['code_presentation_n'] = le_code_presentation.fit_transform(master['code_presentation'])
master['gender_n'] = le_gender.fit_transform(master['gender'])
master['region_n'] = le_region.fit_transform(master['region'])
master['highest_education_n'] = le_highest_education.fit_transform(master['highest_education'])
master['imd_band_n'] = le_imd_band.fit_transform(master['imd_band'])
master['age_band_n'] = le_age_band.fit_transform(master['age_band'])
master['disability_n'] = le_disability.fit_transform(master['disability'])
master['final_result_n'] = le_final_result.fit_transform(master['final_result'])

master = master.drop(['assessment_type', 'code_module', 'code_presentation', 'gender', 'region'
                    , 'highest_education', 'imd_band', 'age_band', 'disability', 'final_result'],axis='columns')

# Remove all rows with NaN data 
master = master.dropna()

scores = []

''' Decision Tree ''' 

# Training dataset based on Distinction, Fail, Pass, Widthdrawn
data = master.drop('final_result_n', axis='columns')
target = master.final_result_n

X = data.iloc[:,] # This select all features in data
y = target

# Discuss preventing overfitting via generalisation (test_train_split)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision tree 
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train,y_train)
all4 = tree_clf.score(X_test,y_test)
all4 # Containing Distinction, Fail, Pass, Withdrawn
print("Score when determining: Distinction, Fail, Pass, Withdraw: {}".format(all4))
scores.append(all4)

#######

# Comparing studentInfo score with Master score 
data_comparison = [] 
data_comparison.append(test_score) 
data_comparison.append(all4) 


types_result = ['studentInfo','All data'] 
data_comparison

bars = plt.bar(types_result, data_comparison)
bars[0].set_color('r')
bars[1].set_color('b')
plt.xlabel('Data') 
plt.ylabel('Model Accuracy')

save_fig("data eval")
plt.clf() 


#######

original_data_edited = master[master.final_result_n != 3] # Remove Distinction
data = original_data_edited.drop('final_result_n', axis='columns')
target = original_data_edited.final_result_n

X = data.iloc[:,] # This select all features in data
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, y_train) 
all3 = tree_clf.score(X_test,y_test)
all3 # Containing Distinction, Fail, Pass
print("Score when determining: Distinction, Fail, Pass: {}".format(all3))
scores.append(all3)

#######

original_data_edited2 = original_data_edited[original_data_edited.final_result_n != 0] # Remove Withdrawn and Distinction
data = original_data_edited2.drop('final_result_n', axis='columns')
target = original_data_edited2.final_result_n

X = data.iloc[:,] # This select all features in data
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, y_train) 
all2 = tree_clf.score(X_test,y_test)
all2 # Fail, Pass
print("Score when determining: Fail, Pass: {}".format(all2))
scores.append(all2)

#######
# Plot a graph to represent how changing what we test for alters the results

types_result = ['D, P, F, W','P, W, F','P, F'] 
scores

bars = plt.bar(types_result, scores)
bars[0].set_color('r')
bars[1].set_color('b')
bars[2].set_color('g')
plt.xlabel('Final Results') 
plt.ylabel('Model Accuracy')

save_fig("final result eval")
plt.clf() 

# Conclude that we will be only testing pass and fail, as our 2nd model is regression, which is a binary classifier

''' Altering Parameters (Decision Tree) ''' 

accuracies_train = []
accuracies_test = [] 
max_depths = []
for i in range(2,10):
    max_depths.append(i)
    
# Altering max_depth parameter 

def test_max_depths(accuracies_train, accuracies_test, max_depths, X_train, X_test, y_train, y_test):
    for i in max_depths:
        tree_clf = DecisionTreeClassifier(max_depth=i, random_state=42)
        tree_clf.fit(X_train, y_train)
        accuracies_train.append(tree_clf.score(X_train, y_train))
        accuracies_test.append(tree_clf.score(X_test, y_test))
    return

test_max_depths(accuracies_train, accuracies_test, max_depths, X_train, X_test, y_train, y_test)

from matplotlib.legend_handler import HandlerLine2D
line1 = plt.plot(max_depths, accuracies_train, label="Train Accuracy")
line2 = plt.plot(max_depths, accuracies_test, label="Test Accuracy")
plt.legend(['Train Accuracy', 'Test Accuracy'])
plt.xlabel('Tree depth') 
plt.ylabel('Model Accuracy')


save_fig("tree depth")
plt.clf() 

# As the blue increases the model gets more overfitted, conclude increasing depth increases overfitting 
print("Optimal max_depth paramter: {}".format(max_depths[accuracies_test.index(max(accuracies_test))]))

#######

accuracies_train = []
accuracies_test = [] 
min_samples_split = [] 
for i in range(2,10):
    min_samples_split.append(i)

# Altering min_samples_split 

def test_min_samples_split(accuracies_train, accuracies_test, min_samples_split, X_train, X_test, y_train, y_test):
    for i in min_samples_split:
        tree_clf = DecisionTreeClassifier(min_samples_split=i, max_depth=3, random_state=42)
        tree_clf.fit(X_train, y_train)
        accuracies_train.append(tree_clf.score(X_train, y_train))
        accuracies_test.append(tree_clf.score(X_test, y_test))
    return

test_min_samples_split(accuracies_train, accuracies_test, min_samples_split, X_train, X_test, y_train, y_test)
line1 = plt.plot(min_samples_split, accuracies_train, label="Train Accuracy")
line2 = plt.plot(min_samples_split, accuracies_test, label="Test Accuracy")
plt.legend(['Train Accuracy', 'Test Accuracy'])
plt.xlabel('Minimum Samples Split') 
plt.ylabel('Model Accuracy')
plt.show

save_fig("min samples split")
plt.clf() 
print("Optimal min_samples_split paramter: {}".format(min_samples_split[accuracies_test.index(max(accuracies_test))]))

#######

accuracies_train = []
accuracies_test = [] 
max_leaf_nodes = [] 
for i in range(2,5):
    max_leaf_nodes.append(i) 

# Altering max_leaf_nodes 

def test_max_leaf_nodes(accuracies_train, accuracies_test, max_leaf_nodes, X_train, X_test, y_train, y_test):
    for i in max_leaf_nodes:
        tree_clf = DecisionTreeClassifier(max_leaf_nodes=i, min_samples_split=2, max_depth=3, random_state=42)
        tree_clf.fit(X_train, y_train)
        accuracies_train.append(tree_clf.score(X_train, y_train))
        accuracies_test.append(tree_clf.score(X_test, y_test))
    return

test_max_leaf_nodes(accuracies_train, accuracies_test, max_leaf_nodes, X_train, X_test, y_train, y_test)
line1 = plt.plot(max_leaf_nodes, accuracies_train, label="Train Accuracy")
line2 = plt.plot(max_leaf_nodes, accuracies_test, label="Test Accuracy")
plt.legend(['Train Accuracy', 'Test Accuracy'])
plt.xlabel('Maximum leaf nodes') 
plt.ylabel('Model Accuracy')
plt.show

save_fig("maximum leaf nodes")
plt.clf() 


# In case of _2 we reach an asymptote relatively early, model isnt overfit as test accuracy remains low 
print("Optimal max_leaf_nodes paramter: {}".format(max_leaf_nodes[accuracies_test.index(max(accuracies_test))]))

#######

accuracies_train = []
accuracies_test = [] 
max_features = []
for i in range(2, len(data.columns)+1):
    max_features.append(i)

# Altering max_features 

def test_max_features(accuracies_train, accuracies_test, max_features, X_train, X_test, y_train, y_test):
    for i in max_features:
        tree_clf = DecisionTreeClassifier(max_features=i, max_leaf_nodes=8, min_samples_split=2, max_depth=3, random_state=42)
        #tree_clf = DecisionTreeClassifier(max_features=i, random_state=42)
        tree_clf.fit(X_train, y_train)
        accuracies_train.append(tree_clf.score(X_train, y_train))
        accuracies_test.append(tree_clf.score(X_test, y_test))
    return

test_max_features(accuracies_train, accuracies_test, max_features, X_train, X_test, y_train, y_test)
line1 = plt.plot(max_features, accuracies_train, label="Train Accuracy")
line2 = plt.plot(max_features, accuracies_test, label="Test Accuracy")
plt.legend(['Train Accuracy', 'Test Accuracy'])
plt.xlabel('Maximum Features') 
plt.ylabel('Model Accuracy')
plt.show

save_fig("maximum features")
plt.clf() 


# Overfitting as we use more Feartures (maybe)
print("Optimal max_features paramter: {}".format(max_features[accuracies_test.index(max(accuracies_test))]))

######

# Pie chart showing relative feature importances 

tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42, min_samples_split=2, max_leaf_nodes=8, max_features=11)
tree_clf.fit(X_train, y_train)
print("Tree score after parameter selection: {}.".format(tree_clf.score(X_test, y_test)))
tree_clf.feature_importances_
feature_importance = dict(zip(X.columns, tree_clf.feature_importances_))

keys = list(feature_importance.keys())
values = list(feature_importance.values())
values_sum = sum(values)
total = 100*values/values_sum

patches, texts = plt.pie(values, startangle=90, radius=1.2)
labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(keys, total)]
plt.legend(patches, labels, loc='center left', bbox_to_anchor=(1, 1.),
           fontsize=8)

save_fig("feature importance pie-chart")
plt.clf() 

''' Tree Visualization ''' 

from graphviz import Source 
from sklearn.tree import export_graphviz


# Explain this score is lower than original because we use a lower max_depth. Higher max_depths = more time 

class_labels = ['Fail', 'Pass']

export_graphviz(
        tree_clf,
        out_file=os.path.join(IMAGES_PATH, "passfail_tree.dot"),
        feature_names=X.columns, 
        class_names=class_labels,
        rounded=True,
        filled=True
    )

Source.from_file(os.path.join(IMAGES_PATH, "passfail_tree.dot"))



''' Hypertuning Parameters (Decision Tree) '''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.model_selection import GridSearchCV
params = {'criterion':['gini','entropy'],'max_depth': [2, 10, 15, 20, 50, 100]}
tree_clf = GridSearchCV(DecisionTreeClassifier(), params, cv=5)
tree_clf.fit(X_train, y_train)

#print(tree_clf.best_params_)

''' Performance Measurement (Decision Tree) ''' 

import seaborn as sns
from sklearn import metrics

# Used a confusion matrix to visualize results 

predictions = tree_clf.predict(X_test)

cm = metrics.confusion_matrix(y_test, predictions)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(tree_clf.score(X_test, y_test))
plt.title(all_sample_title, size = 15)

save_fig('decision tree cm')
plt.clf() 

########
''' ROC Curve (Decision Tree) ''' 

binary_final_result_n = LabelEncoder()
y = binary_final_result_n.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, tree_clf.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, tree_clf.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Decision Tree (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

save_fig('ROC curve DT')
plt.clf() 
# The dotted line represents the ROC curve of a purely random classifier; 
# a good classifier stays as far away from that line as possible (toward the top-left corner).

# Mean squared error
# Indicates the model still not great 
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error for Decision Tree: {}".format(mse))


''' Logistic Regression ''' 

# Training and feature selection 

X = data.iloc[:,] # This select all features in data
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Using Pearson Correlation
plt.figure(figsize=(16,14))
cor = original_data_edited2.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

save_fig("pearson correlation")
plt.clf() 

#Recursive Feature Elimination

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
#Initializing RFE model
rfe = RFE(model, 7)
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  
#Fitting the data to model
model.fit(X_rfe,y)
print(rfe.ranking_)

# Select features via recursive feature elimination and train model 

#X = data.iloc[:,] # This select all features in data
X = data.iloc[:, lambda df: [0,1,2,4,5,9,10]]
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hypertune parameters 

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
lr_clf = GridSearchCV(LogisticRegression(random_state=42), param_grid)
lr_clf.fit(X_train, y_train) 
lr_clf.score(X_test, y_test) 

# Confusion matrix 

# Containing Distinction, Fail, Pass, Withdrawn
predictions = lr_clf.predict(X_test)

cm = metrics.confusion_matrix(y_test, predictions)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(lr_clf.score(X_test, y_test))
plt.title(all_sample_title, size = 15)

save_fig('logistic regression cm')
plt.clf() 

# 1 = pass
# 0 = fail
# Confusion matrix describes the performance of the logistic regression model
# Confunsion matrix shows the model is much better at predicting pass rates than fail rates 


# ROC Curve 

binary_final_result_n = LabelEncoder()
y = binary_final_result_n.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


logit_roc_auc = roc_auc_score(y_test, lr_clf.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, lr_clf.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

save_fig('ROC curve LR')
plt.clf() 


# Mean squared error
 
mean_squared_error(y_test, predictions)





