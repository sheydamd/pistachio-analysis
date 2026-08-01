# =========================
# Imports
# =========================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)


# =========================
# Read Dataset
# =========================

df = pd.read_csv("pistachio/data.csv")


print(df.head())

print("*****************")

print(df.info())

print("*****************")

print("Shape:")
print(df.shape)

print("*****************")

print("Missing Values:")
print(df.isnull().sum())

print("*****************")

print(df.describe())


# =========================
# Remove extra columns
# =========================

if 'id' in df.columns:
    df = df.drop('id', axis=1)


if 'Unnamed: 32' in df.columns:
    df = df.drop('Unnamed: 32', axis=1)



# =========================
# Encode Class
# =========================

encoder = LabelEncoder()

df['Class'] = encoder.fit_transform(
    df['Class']
)


print("\nClasses:")
print(encoder.classes_)



# =========================
# Features / Target
# =========================

X = df.drop(
    'Class',
    axis=1
)

y = df['Class']



# =========================
# Missing Values
# =========================

imputer = SimpleImputer(
    strategy='mean'
)

X = imputer.fit_transform(X)



# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# =========================
# Scaling
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)



# =========================
# KNN
# =========================

knn = KNeighborsClassifier(
    n_neighbors=5
)


knn.fit(
    X_train,
    y_train
)


y_pred = knn.predict(
    X_test
)



# =========================
# Evaluation
# =========================

print("\n====================")
print("KNN Accuracy")
print("====================")

print(
    accuracy_score(
        y_test,
        y_pred
    )
)


print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


print("\nClassification Report")

print(
    classification_report(
        y_test,
        y_pred
    )
)



# =========================
# Find Best K
# =========================

accuracies = []

k_values = range(1,21)


for k in k_values:

    model = KNeighborsClassifier(
        n_neighbors=k
    )

    model.fit(
        X_train,
        y_train
    )


    pred = model.predict(
        X_test
    )


    acc = accuracy_score(
        y_test,
        pred
    )


    accuracies.append(acc)

    print(
        f"K={k} ---> Accuracy={acc}"
    )



# Plot K

plt.figure(figsize=(10,6))


plt.plot(
    k_values,
    accuracies,
    marker='o'
)


plt.xlabel("K")

plt.ylabel("Accuracy")

plt.title(
    "KNN Accuracy"
)

plt.grid(True)

plt.show()



# =========================
# Heatmap
# =========================

numeric_df = df.select_dtypes(
    include=['number']
)


corr = numeric_df.corr()


plt.figure(figsize=(18,12))


sns.heatmap(
    corr,
    cmap="coolwarm"
)


plt.title(
    "Correlation Heatmap"
)


plt.show()



# =========================
# PCA
# =========================

pca = PCA(
    n_components=2
)


X_pca = pca.fit_transform(
    X
)


plt.figure(figsize=(10,7))


plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=y
)


plt.xlabel("PCA 1")

plt.ylabel("PCA 2")


plt.title(
    "PCA Visualization"
)


plt.show()



# =========================
# ROC Curve
# =========================

y_prob = knn.predict_proba(
    X_test
)[:,1]


fpr, tpr, threshold = roc_curve(
    y_test,
    y_prob
)


roc_auc = auc(
    fpr,
    tpr
)



plt.figure(figsize=(8,6))


plt.plot(
    fpr,
    tpr,
    label=f"AUC={roc_auc:.2f}"
)


plt.plot(
    [0,1],
    [0,1],
    linestyle='--'
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)


plt.title(
    "ROC Curve"
)


plt.legend()

plt.show()


print(
    "AUC:",
    roc_auc
)



# =========================
# Compare Models
# =========================

models = {

    "KNN":
    KNeighborsClassifier(),

    "Logistic Regression":
    LogisticRegression(max_iter=1000),

    "SVM":
    SVC(probability=True),

    "Random Forest":
    RandomForestClassifier()

}



print("\nModel Comparison")
print("====================")


for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )


    pred = model.predict(
        X_test
    )


    acc = accuracy_score(
        y_test,
        pred
    )


    print(
        f"{name}: {acc}"
    )