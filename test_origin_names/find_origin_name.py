import pandas as pd
from string import ascii_lowercase
from pandas import read_csv
import numpy as np
import random
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from generate_letter_array import generate_array

#Load datasets
url_ch = "/home/vince/Projects/MachineLearning/MLprojects/test_origin_names/datasets/chinese.txt"
url_fr = "/home/vince/Projects/MachineLearning/MLprojects/test_origin_names/datasets/french.txt"
url_ru = "/home/vince/Projects/MachineLearning/MLprojects/test_origin_names/datasets/russian.txt"
url_jp = "/home/vince/Projects/MachineLearning/MLprojects/test_origin_names/datasets/japanese.txt"

f_ch = open(url_ch, 'r')
f_fr = open(url_fr, 'r')
f_ru = open(url_ru, 'r')
f_jp = open(url_jp, 'r')

#The longest name of all datasets (so the 2D array aren't too big and full of zeros)
longest_name_length = 0

# Putting all names in arrays and counting the number of names in each files
names_ch = []
count_ch = 0
for name in f_ch:
    names_ch.append(name.strip())
    count_ch += 1
    if len(name.strip()) > longest_name_length:
        longest_name_length = len(name.strip())
names_fr = []
count_fr = 0
for name in f_fr:
    names_fr.append(name.strip())
    count_fr += 1
    if len(name.strip()) > longest_name_length:
        longest_name_length = len(name.strip())
names_ru = []
count_ru = 0
for name in f_ru:
    names_ru.append(name.strip())
    count_ru += 1
    if len(name.strip()) > longest_name_length:
        longest_name_length = len(name.strip())
names_jp = []
count_jp = 0
for name in f_jp:
    names_jp.append(name.strip())
    count_jp += 1
    if len(name.strip()) > longest_name_length:
        longest_name_length = len(name.strip())
        
print("Longest name is", longest_name_length,"characters long")        
 
# Setting the max amount of names to use for training and testing
max_number_names = 80 #train + test
training_ratio = 0.8 #ratio to use for training
training_sample = int(np.floor(max_number_names*training_ratio))
testing_sample = int(max_number_names - training_sample)

# Make sure the number asked is bigger than the smaller datasets
if max_number_names > min(count_ch, count_fr, count_ru, count_jp):
    print("**** You asked for more names than there is in the smallest dataset...")
    raise Exception("**** Exiting ****")

# Randomly populate the training and test datasets
random.shuffle(names_ch)
random.shuffle(names_fr)
random.shuffle(names_ru)
random.shuffle(names_jp)

#Makes new 2D array with the origin as the second column
arr_names_ch = []
for i in range(0,max_number_names):
    arr_names_ch.append(str(names_ch[i]))
arr_names_fr = []
for i in range(0,max_number_names):
    arr_names_fr.append(str(names_fr[i]))
arr_names_ru = []
for i in range(0,max_number_names):
    arr_names_ru.append(str(names_ru[i]))
arr_names_jp = []
for i in range(0,max_number_names):
    arr_names_jp.append(str(names_jp[i]))
   
# Create dataframes    
df_ch = pd.DataFrame(data=arr_names_ch, columns=["name"])
df_fr = pd.DataFrame(data=arr_names_fr, columns=["name"])
df_ru = pd.DataFrame(data=arr_names_ru, columns=["name"])
df_jp = pd.DataFrame(data=arr_names_jp, columns=["name"])
#print(df_ch.head(10))
#print(df_ch.shape)

#Adds a column (with the name as a 2D array of letters) in the dataframes
# Using pandas series since using lists was created nest lists of arrays of arrays
names_2d_array_ch = pd.Series([])
for index, row in df_ch.iterrows():
    names_2d_array_ch[index] = generate_array(row['name'],longest_name_length)
df_ch.insert(1, "letters_array", names_2d_array_ch, True)
names_2d_array_fr = pd.Series([])
for index, row in df_fr.iterrows():
    names_2d_array_fr[index] = generate_array(row['name'],longest_name_length)
df_fr.insert(1, "letters_array", names_2d_array_fr, True)
names_2d_array_ru = pd.Series([])
for index, row in df_ru.iterrows():
    names_2d_array_ru[index] = generate_array(row['name'],longest_name_length)
df_ru.insert(1, "letters_array", names_2d_array_ru, True)
names_2d_array_jp = pd.Series([])
for index, row in df_jp.iterrows():
    names_2d_array_jp[index] = generate_array(row['name'],longest_name_length)
df_jp.insert(1, "letters_array", names_2d_array_jp, True)

# Adds the origin columns
df_ch.insert(2, "origin", "Chinese", True)
df_fr.insert(2, "origin", "French", True)
df_ru.insert(2, "origin", "Russian", True)
df_jp.insert(2, "origin", "Japanese", True)

#print(df_ch.head(10))
#print(df_ch.shape)

# Merge all dataframes into one and shuffle
frames = [df_ch, df_fr, df_ru, df_jp]
df_all = pd.concat(frames, ignore_index=True)
df_all = shuffle(df_all)

print(df_all)
print(df_all.shape)
print(df_all.groupby('origin').size())

# Now split dataset into train and test
#X = pd.DataFrame(df_all, columns=['letters_array'])
#y = pd.DataFrame(df_all, columns=['origin'])
X = df_all.loc[:,'letters_array'].to_numpy()
y = df_all.loc[:,'origin']

# writes the X array into a numpy 2D array
arr2d = np.zeros((len(X),len(X[0])))
for i in range(0,len(X)):
    for j in range(0,len(X[0])):
        arr2d[i][j] = X[i][j]

X_train, X_validation, Y_train, Y_validation = train_test_split(arr2d, y, test_size=1-training_ratio, shuffle=False)

#Keep info on the validation names
X_validation_names = pd.DataFrame(df_all, columns=['name'])
X_validation_names = X_validation_names[X_validation_names.index.isin(Y_validation.index)]

#print(arr2d.shape)
#print(y.shape)
#print(X_train.shape)
#print(Y_train.shape)

#Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


#Make predictions on validation dataset - Fit part
model = LogisticRegression(solver='liblinear', multi_class='ovr')
model.fit(X_train, Y_train)

# Make predictions on validation dataset
predictions = model.predict(X_validation)

# Loops over the validation dataset so you can see how good/bad the predictions are
for i in range(0,len(predictions)):
    print("Name: " + str(X_validation_names.iloc[i]['name']))
    print("Origin: " + str(Y_validation.iloc[i][:]))
    print("Prediction: " + str(predictions[i]))
    print("-----------------")

#evaluate predictions
print("Accuracy score:")
print(accuracy_score(Y_validation, predictions))
print("Confusion matrix:")
print(confusion_matrix(Y_validation, predictions))
print("Classification report:")
print(classification_report(Y_validation, predictions))

# here's the part where you can manually try names
manual_name="Gauthier"
if len(manual_name) > longest_name_length:
    raise Exception("**** Name too long! ****")

manual_X_validation = generate_array(manual_name,longest_name_length)
manual_X_validation = np.reshape(manual_X_validation, (1, -1))
manual_prediction = model.predict(manual_X_validation)
print("The name", manual_name,"seems to be",str(manual_prediction))
