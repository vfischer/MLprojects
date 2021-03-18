import pandas as pd
import csv
import numpy as np
from pandas import read_csv
from string import ascii_lowercase

# Dictionaries for all letters and special case (spaces and special characters)
letter_dict = {" ": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8, "i": 9, "j": 10, "k": 11, "l": 12, "m": 13, 
			   "n": 14, "o": 15, "p": 16, "q": 17, "r": 18, "s": 19, "t": 20, "u": 21, "v": 22, "w": 23, "x": 24, "y": 25, "z": 26, "-": 27
}

def generate_array(name, max_name_length):
    #print(name)
    
    # Array the name will be put into
    name_arr = []
    
    # Name is now all lower case
    name = name.lower()
    
    # Loop on all letters and change special characters into - (keep ascii letters and spaces)
    for letter in name:
        if letter not in ascii_lowercase and letter != " ":
            name_arr += "-"
        else:
            name_arr += letter
            
    # Fills the rest of the name with empty spaces so that the name length is equal to max_name_length
    if len(name_arr) < max_name_length:
        name_arr += (max_name_length-len(name_arr))*" "
    
    #print(name_arr)
    
    # Initialize 2D array (row: "letters from space, a,b,...to z, -", column: location in the word, max number of columns is max_name_length)
    arr_letters = np.zeros((len(letter_dict),max_name_length))
    
    
    for letter in range(0,max_name_length):
        #print(letter_dict[name_arr[letter]])
        for dict_i in range(0,len(letter_dict)):
            if letter_dict[name_arr[letter]] == dict_i:
                arr_letters[dict_i][letter] = 1
    
    #print(arr_letters) 
    
    # Returns the array (shape is (28,max_name_length))   (28 is 26 characters + space + special)
    return arr_letters
