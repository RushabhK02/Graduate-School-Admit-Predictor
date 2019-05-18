# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('Admission_Predict_Ver1.1.csv')
dataset = dataset.iloc[:,1:]
print('Data Show Describe\n')
dataset.describe()

#Variance plot
dataset.columns = dataset.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
sns.pairplot(dataset,hue="chance_of_admit")
plt.show()

#Data distribution plot
for i,col in enumerate(dataset.columns.values):
    plt.subplot(5,3,i+1)
    plt.scatter([i for i in range(500)],dataset[col].values.tolist())
    plt.title(col)
    fig,ax=plt.gcf(),plt.gca()
    fig.set_size_inches(10,10)
    plt.tight_layout()
plt.show()

X = dataset.iloc[:, : -1].values
y = dataset.iloc[:, [-1]].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Correlation plot
train_data = pd.DataFrame(X_train.copy(), columns = ['gre', 'toefl', 'university_ranking', 'sop', 'lor', 'grade', 'researchexp'])
train_data['chance_of_admit'] = y_train
C_mat = train_data.corr()
fig = plt.figure(figsize = (15,15))
sns.heatmap(C_mat, vmax = .8, square = True)
plt.show()

#Removing researchexp due to lower correlation with admit chance
#X_train = X_train[:,:-1]
#X_test = X_test[:,:-1]

#PCA Analysis
from sklearn.decomposition import PCA
pca=PCA().fit(X_train)
print(pca.explained_variance_ratio_)

# Fitting classifier to the Training set
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(activation="relu", input_dim=7, units=4, kernel_initializer="normal"))

# Adding the output layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="normal"))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'logcosh', metrics = ['accuracy', 'mean_absolute_error'])
classifier.summary()

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred_binary = (y_pred>0.5)
y_test_binary = (y_test>0.5)

# Predicting the Test set results
from sklearn.metrics import roc_curve, auc
y_pred = classifier.predict(X_test)
y_proba=classifier.predict_proba(X_test)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test_binary,y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_binary, y_pred_binary)
