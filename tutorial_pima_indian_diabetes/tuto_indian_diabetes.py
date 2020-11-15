from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
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
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
names = ['nb_pregancy', 'plasma_gluc', 'blood_press', 'skin_fold_thick', 'insulin', 'bmi', 'diab_pedigree', 'age', 'diab_positive']
dataset = read_csv(url, names=names)

#shape
print(dataset.shape)
#head
print(dataset.head(10))
#descriptions
print(dataset.describe())
#class distribution
print(dataset.groupby('diab_positive').size())

#box and whisker plot
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#pyplot.show()

#histograms
#dataset.hist()
#pyplot.show()

#scatter plot scatter_matrix
#scatter_matrix(dataset)
#pyplot.show()

#SPlit validation dataset
array = dataset.values
X = array[:,0:8]
y = array[:,8]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

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
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# comapre algos    
#pyplot.boxplot(results, labels=names)
#pyplot.title('Algorithm Comparison')
#pyplot.show()

#Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print(X_validation)
print(Y_validation)
#evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
