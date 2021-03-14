from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#Load dataset
url_red = "/home/vince/MachineLearning/test_regression_wine_dataset/winequality-red.csv"
url_white = "/home/vince/MachineLearning/test_regression_wine_dataset/winequality-white.csv"
names = ['fix_acidity', 'vol_acidity', 'citric_acid', 'sugar', 'chlor', 'free_SO2', 'tot_S02', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
dataset_red = read_csv(url_red, sep=';', names=names, header=0)
dataset_white = read_csv(url_white, sep=';', names=names, header=0)

#shape
print(dataset_red.shape)
#head
print(dataset_red.head(10))
#descriptions
print(dataset_red.describe())
#class distribution
print(dataset_red.groupby('quality').size())

correlations = dataset_red.corr()['quality'].drop('quality')
print(correlations)

#box and whisker plot
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#pyplot.show()

#histograms
#dataset_red.hist()
#pyplot.show()

#scatter plot scatter_matrix
#scatter_matrix(dataset_red)
#pyplot.show()

#only show high correlation
#def get_features(correlation_threshold):
#    abs_corrs = correlations.abs()
#    high_correlations = abs_corrs
#    [abs_corrs > correlation_threshold].index.values.tolist()
#    return high_correlations

#features = get_features(0.05) 
#print(features) 

#SPlit validation dataset
array_red = dataset_red.values
array_white = dataset_white.values
X_red = array_red[:,0:11]
y_red = array_red[:,11]
X_white = array_white[:,0:11]
y_white = array_white[:,11]

X_train_red, X_validation_red, Y_train_red, Y_validation_red = train_test_split(X_red, y_red, test_size=0.20, random_state=1)
X_train_white, X_validation_white, Y_train_white, Y_validation_white = train_test_split(X_white, y_white, test_size=0.20, random_state=1)

print(X_train_red)
print(Y_train_red)

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
    cv_results = cross_val_score(model, X_train_red, Y_train_red, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# comapre algos    
#pyplot.boxplot(results, labels=names)
#pyplot.title('Algorithm Comparison')
#pyplot.show()

#Make predictions on validation dataset
model_red = LinearDiscriminantAnalysis()
model_red.fit(X_train_red, Y_train_red)

#Make predictions on validation dataset
model_white = LinearDiscriminantAnalysis()
model_white.fit(X_train_white, Y_train_white)

# this gives the coefficients of the 10 features selected above.  
#print(model_red.coef_)

# this gives the coefficients of the 10 features selected above.  
# print(model_white.coef_)

#save model to a file
#filename = 'saved_model.sav'
#pickle.dump(model_red,open(filename, 'wb'))

#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_validation,Y_validation)
predictions_train_red = model_red.predict(X_train_red)
predictions_train_white = model_white.predict(X_train_white)
predictions_test_red = model_red.predict(X_validation_red)
predictions_test_white = model_white.predict(X_validation_white)

# calculating rmse
train_red_rmse = mean_squared_error(predictions_train_red, Y_train_red) ** 0.5
print(train_red_rmse)
test_red_rmse = mean_squared_error(predictions_test_red, Y_validation_red) ** 0.5
print(test_red_rmse)# rounding off the predicted values for test set
predicted_data_red = np.round_(test_red_rmse)
print(predicted_data_red)
print('Mean Absolute Error:', mean_absolute_error(Y_validation_red, predictions_test_red))
print('Mean Squared Error:', mean_squared_error(Y_validation_red, predictions_test_red))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(Y_validation_red, predictions_test_red)))
# displaying coefficients of each feature
coefficients_red = pd.DataFrame(model_red.coef_)
#coefficients_red.columns = ['Coefficient'] 
print(coefficients_red)

#print(X_validation)
#print(Y_validation)
#evaluate predictions
#print("Red wine:")
#print(accuracy_score(Y_validation_red, predictions_red))
#print(confusion_matrix(Y_validation_red, predictions_red))
#print(classification_report(Y_validation_red, predictions_red))
#print("White wine:")
#print(accuracy_score(Y_validation_white, predictions_white))
#print(confusion_matrix(Y_validation_white, predictions_white))
#print(classification_report(Y_validation_white, predictions_white))
