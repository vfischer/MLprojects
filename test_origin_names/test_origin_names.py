import pandas as pd
from string import ascii_lowercase
from pandas import read_csv
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

#Load dataset
url = "/home/vince/MachineLearning/test_origin_names/name_origin_occurences_letters.csv"
#names = ['name', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'origin']
dataset = read_csv(url, sep=',', header=0)

#Load test dataset for one shot prediction
url_oneshot = "/home/vince/MachineLearning/test_origin_names/name_origin_occurences_letters_oneshot_validation.csv"
#names = ['name', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'origin']
dataset_oneshot = read_csv(url_oneshot, sep=',', header=0)

#shuffle dataset to mix origins
dataset = shuffle(dataset)
#shape
print(dataset.shape)
#head
print(dataset.head(10))
#descriptions
#print(dataset.describe())
#class distribution
#print(dataset.groupby('origin').size())

#box and whisker plot
#dataset.plot(kind='box', subplots=True, layout=(26,26), sharex=False, sharey=False)
#pyplot.show()

#histograms
#dataset.hist()
#pyplot.show()

#scatter plot scatter_matrix
#scatter_matrix(dataset)
#pyplot.show()

#SPlit validation dataset
#array = dataset.values
#X_validation_names = array[:,0]
#X = array[:,1:27]
#y = array[:,27]
X_validation_names = pd.DataFrame(dataset, columns=['name'])
X = pd.DataFrame(dataset, columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
y = pd.DataFrame(dataset, columns=['origin'])

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.10, shuffle=False)

X_validation_names = X_validation_names[X_validation_names.index.isin(Y_validation.index)]


#Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results =[]
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# comapre algos    
#pyplot.boxplot(results, labels=names)
#pyplot.title('Algorithm Comparison')
#pyplot.show()

#Make predictions on validation dataset - Fit part
model = SVC(gamma='auto')
model.fit(X_train, Y_train)

#Cooment those lines if you want to use the validation dataet from sklearn (not a file you inputted)
#X_validation_names= pd.DataFrame(dataset_oneshot, columns=['name'])
#X_validation = pd.DataFrame(dataset_oneshot, columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
#Y_validation = pd.DataFrame(dataset_oneshot, columns=['origin'])

# Make predictions on validation dataset
predictions = model.predict(X_validation)

#print(len(X_validation_names))
#print(len(Y_validation))
#print(len(predictions))
#print("X validation names:")
#print(X_validation_names)
#print("Y validation:")
#print(Y_validation)
#print("Predictions:")
#print(predictions)

for i in range(len(predictions)):
    print("Name: " + str(X_validation_names.iloc[i]['name']))
    print("Origin: " + str(Y_validation.iloc[i]['origin']))
    print("Prediction: " + str(predictions[i]))
    print("-----------------")
    
#evaluate predictions
print("Accuracy score:")
print(accuracy_score(Y_validation, predictions))
print("Confusion matrix:")
print(confusion_matrix(Y_validation, predictions))
print("Classification report:")
print(classification_report(Y_validation, predictions))
