# STEP #1: IMPORTING LIBRARIES

import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import seaborn as sns # Import seaborn for statistical data visualization

# STEP #2: IMPORT THE DATASET  

iris_df = pd.read_csv('Iris.csv')

# scatterplot of SepalLengthCm versus SepalWidthCm
sns.scatterplot( x = 'SepalLengthCm', y = 'SepalWidthCm', hue = 'Species', data = iris_df)

# scatterplot of PetalLengthCm versus PetalWidthCm
sns.scatterplot(x = 'PetalLengthCm', y = 'PetalWidthCm', hue = 'Species', data = iris_df)

# Let's show the Violin plot 
plt.figure(figsize=(12,12))

plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=iris_df)

plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=iris_df)

plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=iris_df)

plt.subplot(2,2,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=iris_df)

# Let's try the Seaborn pairplot 
sns.pairplot(iris_df, hue = 'Species')

# Let's check the correlation between the variables 
plt.figure(figsize=(12,9)) 
sns.heatmap(iris_df.corr(),annot=True) 

# STEP #3: DATA CLEANING

# Let's drop the Species coloumn
X = iris_df.drop(['Species'],axis=1)

# Let's take our target class
y = iris_df['Species']

# Import train_test_split from scikit library
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35,  random_state = 10)

# STEP #4: TRAINING THE MODEL

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 6 , metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# STEP #5: EVALUATING THE MODEL

# Let's check our model using confusion matrix and classification report
from sklearn.metrics import classification_report, confusion_matrix
y_predict = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True, fmt="d")



















