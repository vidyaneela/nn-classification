# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Import the necessary packages & modules

### STEP 2:
Load and read the dataset

### STEP 3:
Perform pre processing and clean the dataset

### STEP 4:
 Encode categorical value into numerical values using ordinal/label/one hot encoding
 
 ### STEP 5:
 Visualize the data using different plots in seaborn
 
 ### STEP 6:
 Normalize the values and split the values for x and y
 
 ### STEP 7:
 Build the deep learning model with appropriate layers and depth
 
 ### STEP 8:
 Analyze the model using different metrics

### STEP 9:
 Plot a graph for Training Loss, Validation Loss Vs Iteration & for Accuracy, Validation Accuracy vs Iteration

### STEP 10:
 Save the model using pickle

### STEP 11:
 Using the DL model predict for some random inputs

## PROGRAM

```
Developed By: VIDYA NEELA M
Reg No:212221230120
```

###import libraries
```
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pylab as plt
```
```
import pandas as pd
customer_df=pd.read_csv('/content/customers (1).csv')
customer_df.columns
from google.colab import drive
drive.mount('/content/drive')
customer_df.dtypes
customer_df.shape
customer_df.isnull().sum()
customer_df_cleaned = customer_df.dropna(axis=0)
customer_df_cleaned.isnull().sum()
customer_df_cleaned.shape
customer_df_cleaned.dtypes
customer_df_cleaned['Gender'].unique()
customer_df_cleaned['Ever_Married'].unique()
customer_df_cleaned['Var_1'].unique()
customer_df_cleaned['Spending_Score'].unique()
customer_df_cleaned['Graduated'].unique()
customer_df_cleaned['Profession'].unique()
customer_df_cleaned['Segmentation'].unique()
from sklearn.preprocessing import OrdinalEncoder
categories_list=[['Male', 'Female'],
                 ['No', 'Yes'],
                 ['No', 'Yes'],
                 ['Healthcare', 'Engineer', 'Lawyer', 'Artist', 'Doctor','Homemaker', 'Entertainment', 'Marketing', 'Executive'],
                 ['Low', 'High', 'Average']
                 ]
enc=OrdinalEncoder(categories=categories_list)
customer_l=customer_df_cleaned.copy()
customer_l[['Gender',
            'Ever_Married',
            'Graduated',
            'Profession',
            'Spending_Score']]=enc.fit_transform(customer_l[['Gender',
                                                              'Ever_Married',
                                                              'Graduated',
                                                              'Profession',
                                                              'Spending_Score']])
le=LabelEncoder()
customer_l.dtypes
customer_l['Segmentation'] = le.fit_transform(customer_l['Segmentation'])
customer_l=customer_l.drop('ID',axis=1)
customer_l=customer_l.drop('Var_1',axis=1)
```
# Calculate the correlation matrix
```
corr = customer_l.corr()
```
# Plot the heatmap
```
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="BuPu",
        annot= True)
```
```
sns.pairplot(customer_l)
sns.distplot(customer_l['Age'])
plt.figure(figsize=(10,6))
sns.countplot(customer_l['Family_Size'])

plt.figure(figsize=(10,6))
sns.boxplot(x='Family_Size',y='Age',data=customer_l)
plt.figure(figsize=(10,6))
sns.scatterplot(x='Family_Size',y='Spending_Score',data=customer_l)
plt.figure(figsize=(10,6))
sns.scatterplot(x='Family_Size',y='Age',data=customer_l)
customer_l.describe()
customer_l['Segmentation'].unique()
X=customer_l[['Gender',
            'Ever_Married',
            'Age',
            'Graduated',
            'Profession','Work_Experience',
            'Spending_Score','Family_Size',]].values
y1=customer_l[['Segmentation']].values
y1[10]
one_hot_enc=OneHotEncoder()
one_hot_enc.fit(y1)
y1.shape
y = one_hot_enc.transform(y1).toarray()
y.shape
y1[0]
X.shape
X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.33,
                                               random_state=50)
X_train[0]
X_train.shape
scaler_age = MinMaxScaler()
scaler_age.fit(X_train[:,2].reshape(-1,1))
X_train_scaled = np.copy(X_train)
X_test_scaled = np.copy(X_test)
# To scale the Age column
X_train_scaled[:,2] = scaler_age.transform(X_train[:,2].reshape(-1,1)).reshape(-1)
X_test_scaled[:,2] = scaler_age.transform(X_test[:,2].reshape(-1,1)).reshape(-1)
# Creating the model
ai_brain=Sequential([
    Dense(10,input_shape=(8,)),
    Dense(12,activation='relu'),
    Dense(4,activation='softmax')
])
ai_brain.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=2)
ai_brain.fit(x=X_train_scaled,y=y_train,
             epochs=2000,batch_size=256,
             validation_data=(X_test_scaled,y_test),
             )
metrics = pd.DataFrame(ai_brain.history.history)
metrics.head()
metrics[['loss','val_loss']].plot()
# Sequential predict_classes function is deprecated
# predictions = ai_brain.predict_classes(X_test)
x_test_predictions = np.argmax(ai_brain.predict(X_test_scaled), axis=1)
x_test_predictions.shape
y_test_truevalue = np.argmax(y_test,axis=1)
y_test_truevalue.shape
print(confusion_matrix(y_test_truevalue,x_test_predictions))
print(classification_report(y_test_truevalue,x_test_predictions))
# Saving the Model
ai_brain.save('customer_classification_model.h5')
# Saving the data
with open('customer_data.pickle', 'wb') as fh:
   pickle.dump([X_train_scaled,y_train,X_test_scaled,y_test,customer_l,customer_df_cleaned,scaler_age,enc,one_hot_enc,le], fh)
# Loading the Model
ai_brain = load_model('customer_classification_model.h5')
   [X_train_scaled,y_train,X_test_scaled,y_test,customers_1,customer_df_cleaned,scaler_age,enc,one_hot_enc,le]=pickle.load(fh)
x_single_prediction = np.argmax(ai_brain.predict(X_test_scaled[1:2,:]), axis=1)
print(x_single_prediction)
print(le.inverse_transform(x_single_prediction))
```
## Dataset Information

![image](https://user-images.githubusercontent.com/94169318/227700189-27db199f-2e96-4da0-bb4f-7e4e2f03d480.png)


## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://user-images.githubusercontent.com/94169318/227700216-ea649440-6c8c-438a-b86d-7a0e5ba1d2e8.png)


### Classification Report

![image](https://user-images.githubusercontent.com/94169318/227700242-51b1109b-0c30-40a0-b2e3-080146bc415a.png)


### Confusion Matrix

![image](https://user-images.githubusercontent.com/94169318/227700265-d358f5ea-5a8c-4225-a5d5-3f331175690b.png)



### New Sample Data Prediction

![image](https://user-images.githubusercontent.com/94169318/227700303-6a7f01b2-a8cf-406c-9276-f513ba36cd29.png)


## RESULT

Thus a neural network classification model for the given dataset is written and executed successfully.
