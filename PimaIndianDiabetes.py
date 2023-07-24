import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler, Binarizer
from sklearn.linear_model import LogisticRegression

diabetes_data = pd.read_csv("diabetes.csv")
print("\nDiabetes Dataframe :\n", diabetes_data.head().to_string())
print("\nOutcome counts of Diabetes :\n", diabetes_data["Outcome"].value_counts())

definition_dictionary = {
    "Pregnancies": "Number of Pregnancy",
    "Glucose": "Glucose test figures",
    "BloodPressure": "Blood Pressure in mm Hg",
    "SkinThickness": "Thickness of skin behind Triceps in mm",
    "Insulin": "Blood Insulin level in mu U/ml",
    "BMI": "Body Mass Index",
    "DiabetesPedigreeFunction": "Diabetes Pedigree Figures",
    "Age": "Age",
    "Outcome": "Class Value(0 or 1)"
}

print(diabetes_data.info())


def get_clf_eval(y_test, pred=None, pred_proba=None):
    print("\n### Evaluation Indexes ###\n")
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc = roc_auc_score(y_test, pred_proba)
    print("Confusion Matrix :\n", confusion)
    print("Accuracy : {0:.4f}".format(accuracy) +
          "\nPrecision : {0:.4f}".format(precision) +
          "\nrecall : {0:.4f}".format(recall) +
          "\nf1 : {0:.4f}".format(f1) +
          "\nroc auc : {0:.4f}".format(roc))


def precision_recall_curve_plot(y, pred_prob):
    precisions, recalls, thresholds = precision_recall_curve(y, pred_prob)

    plt.figure(figsize=(8,6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle="--", label="precision")
    plt.plot(thresholds, recalls[0:threshold_boundary], label="recall")

    # X-axis(threshold)'s scale unit to 0.1
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, .1), 2))

    # Setting X-axis, y-axis, label and legend
    plt.xlabel("Threshold value")
    plt.ylabel("Precision and Recall value")
    plt.legend()
    plt.grid()
    plt.show()


X = diabetes_data.iloc[:, :-1]
y = diabetes_data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=156, stratify=y)

lr_clf = LogisticRegression(solver="liblinear")
lr_clf.fit(X_train, y_train)
pred = lr_clf.predict(X_test)
pred_proba = lr_clf.predict_proba(X_test)[:, 1]

get_clf_eval(y_test, pred, pred_proba)
precision_recall_curve_plot(y_test, pred_proba)

print(diabetes_data.describe().to_string())
plt.hist(diabetes_data["Glucose"], bins=100)
plt.show()

# Calcuate Number of 0s and percentage from the features with 0 value.
# "Pregnancy" is neglected since 0 pregnancy is a valid case
zero_features = ["Glucose", "BloodPressure", "Insulin", "BMI", "SkinThickness"]
total_count = diabetes_data["Glucose"].count()
for feature in zero_features:
    zero_count = diabetes_data[diabetes_data[feature] == 0][feature].count()
    print("\n{0} feature's 0 counts are {1}, and percentage is {2:.2f}".format(feature, zero_count, 100 * zero_count / total_count))

# Substitute 0s with average values of each column
diabetes_data[zero_features] = diabetes_data[zero_features].replace(0, diabetes_data[zero_features].mean())

# Scale the whole feature data using StandardScaler, Train/predict data using the substituted data
X = diabetes_data.iloc[:, :-1]
y = diabetes_data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=156, stratify=y)

lr_clf = LogisticRegression(solver="liblinear")
lr_clf.fit(X_train, y_train)
pred = lr_clf.predict(X_test)
pred_proba = lr_clf.predict_proba(X_test)[:, 1]

get_clf_eval(y_test, pred, pred_proba)
precision_recall_curve_plot(y_test, pred_proba)


def get_clf_eval_by_threshold(y_test, pred_proba, thresholds):
    for th in thresholds:
        print("\nThreshold value :", np.round(th, 2))
        binarizer = Binarizer(threshold=th).fit(pred_proba)
        pred = binarizer.transform(pred_proba)
        get_clf_eval(y_test, pred, pred_proba)


thresholds = [.3, .33, .36, .39, .42, .45, .48, .5]
pred_proba = lr_clf.predict_proba(X_test)
get_clf_eval_by_threshold(y_test, pred_proba[:,1].reshape(-1,1), thresholds)

binarizer = Binarizer(threshold=0.48)
pred_th_048 = binarizer.fit_transform(pred_proba[:,1].reshape(-1, 1))

print("\n### ### ### ### ### ### ### ### ### ### ### ### ###")

get_clf_eval(y_test, pred_th_048, pred_proba[:, 1])