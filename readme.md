# Pistachio Classification

This project is a simple machine learning experiment where we try to classify pistachios into different classes based on their measured features.

The main idea is not too complicated: give the models some information about a pistachio and see if they can figure out which class it belongs to.

I also tried a few different models instead of relying on just one, so we can get a better idea of what works well for this dataset.

# Dataset

The dataset is stored in:

pistachio/data.csv

It contains different numerical features describing pistachio samples, and the column we're trying to predict is "Class".

Before doing anything with the models, the data is cleaned up a little:

- Unnecessary columns are removed.
- Missing values are replaced with the mean of the feature.
- The class labels are converted into numbers using "LabelEncoder".
- The data is split into 80% training and 20% testing.
- Features are standardized using "StandardScaler".

The idea is to let the models learn from the training data and then test them on samples they haven't seen before.

# Models

I used four different classification models:

### KNN

KNN looks at the samples closest to a new sample and uses their classes to make a prediction.

I started with "K=5", but instead of assuming that 5 is the best choice, I tested values from 1 to 20 and plotted their accuracies.

This makes it easier to see how changing "K" affects the model.

### Logistic Regression

A simple classification model that works well as a baseline.

It's useful here because it gives us something relatively straightforward to compare with the other models.

### SVM

SVM tries to find a good decision boundary between the different classes.

It's another useful option when the classes can be separated based on the available features.

### Random Forest

Random Forest combines a bunch of decision trees and uses them together to make the final prediction.

It's a nice model to include because it works quite differently from KNN, Logistic Regression, and SVM.


# How the Models Are Evaluated

The main metric used for comparing the models is Accuracy.

Basically, it tells us:

«"Out of all the predictions, how many did the model get right?"»

For the KNN model, I also used:

- Confusion Matrix to see where the predictions were correct or incorrect.
- Classification Report for Precision, Recall, and F1-score.
- ROC Curve and AUC to get another view of how well the model separates the classes.

Using a few different metrics makes the evaluation a little more meaningful than looking at accuracy alone.


## Visualizations

There are also two visual parts in the project.

### Correlation Heatmap

The heatmap shows how the numerical features are related to each other.

This is useful for getting a quick idea of which features tend to move together.

### PCA

PCA reduces all the features down to just two components.

This doesn't mean we're replacing the original data with only two features for the models. It's mainly being used for visualization, so we can get a rough idea of how the samples and classes are distributed.

# Libraries

The project mainly uses:

- Pandas for reading and working with the dataset.
- NumPy for numerical operations.
- Matplotlib for creating the plots.
- Seaborn for the correlation heatmap.
- Scikit-learn for preprocessing, PCA, classification models, and evaluation.
