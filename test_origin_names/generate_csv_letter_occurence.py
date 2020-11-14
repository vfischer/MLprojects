import pandas as pd
import csv
from pandas import read_csv
from string import ascii_lowercase

#Load dataset
url = "/home/vince/MachineLearning/test_origin_names/list_first_names.csv"
names = ['name', 'origin']
dataset = read_csv(url, names=names, delimiter=',', header=None)

#shape
print(dataset.shape)
#head
print(dataset.head(5))
#descriptions
#print(dataset.describe())
#class distribution
#print(dataset.groupby('origin').size())

print(dataset['name'])
print(dataset['name'][3])
print(dataset['origin'][3])
print(len(dataset['name']))

with open('name_origin_occurences_letters.csv', 'w', newline='') as csvfile_out:
    namewriter = csv.writer(csvfile_out, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    namewriter.writerow(['name']+ ['a'] + ['b'] + ['c'] + ['d'] + ['e'] + ['f'] + ['g'] + ['h'] + ['i'] + ['j'] + ['k'] + ['l'] + ['m'] + ['n'] + ['o'] + ['p'] + ['q'] + ['r'] + ['s'] + ['t'] + ['u'] + ['v'] + ['w'] + ['x'] + ['y'] + ['z'] + ['origin'])
    for i in range(len(dataset['name'])):
        name_to_test = dataset['name'][i]
        counter = []
        for c in ascii_lowercase:
            counter.append(name_to_test.lower().count(c))
        namewriter.writerow([dataset['name'][i]] + counter + [dataset['origin'][i]])
