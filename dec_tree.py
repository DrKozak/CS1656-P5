#dec_tree.py 

import argparse
import collections
import csv
import json
import glob
import math 
import numpy
import os
import pandas as panda
import re
import requests
import string
import sys
import time

def read_input(t_input): 
    #Read tree text from the file
    with open(t_input) as t_file:
        get_lines = t_file.readlines() 
        #read until EOF 
        get_lines = [x.rstrip() for x in get_lines if x.rstrip()]
        #Remove newline characters
    return get_lines

def parse_tree(get_info):
    #Given a line in the tree, parse for its attributes including depth of tree, feat, value, classification, classification count
    tokenizer = get_info.split(" ")
    tokenizer = [x for x in tokenizer if x]
    pipes = tokenizer.count("|")
    #The number of pipes indicates the depth
    feat = tokenizer[pipes]
    feat_value = tokenizer[pipes+1].strip(":")
    classification = None 
    #Final classification
    classification_count = -1
    if ( len(tokenizer) - pipes ) == 4: 
        # classification found
        classification = tokenizer[pipes+2]
        classification_count_tokenizer = tokenizer[pipes+3]
        classification_count = int(classification_count_tokenizer[1:len(classification_count_tokenizer)-1]) 
        #get number or count, within the two () at the front and end
    return (pipes, feat, feat_value, classification, classification_count) 
    

def get_tree(tree_text): 
    #make the tree in a dictionary
    tree = {}
    first_lvl = (None, None) 
    #Keep track of the path that we've travelled down in the tree for the first traversal
    second_lvl = (None, None)
    #Keep track of the path that we've travelled down in the tree for the second traversal
    decider = set([])
     #All possible classifications in the tree, so good or bad
    for get_info in tree_text:
        depth, feat, feat_value, classification, classification_count = parse_tree(get_info) 
        # Get the line info
        if depth == 0:
            if not feat in tree:
                tree[feat] = {}
                tree[feat][feat_value] = {}
            else:
                if not feat_value in tree[feat]:
                    tree[feat][feat_value] = {}

            if not classification is None:
                tree[feat][feat_value][classification] = 0
                 #num occurrences, int to 0
                decider.add(classification) 

            first_lvl = (feat, feat_value)
            second_lvl = (None, None) 
            #Reset just in case we went back up the decision tree rather than down
        elif depth == 1:
            first_lvl_feat = first_lvl[0]
            first_lvl_val = first_lvl[1]
            if not feat in tree[first_lvl_feat][first_lvl_val]:
                tree[first_lvl_feat][first_lvl_val][feat] = {}
                tree[first_lvl_feat][first_lvl_val][feat][feat_value] = {}
            else:
                if not feat_value in tree[first_lvl_feat][first_lvl_val][feat]:
                    tree[first_lvl_feat][first_lvl_val][feat][feat_value] = {}

            if not classification is None:
                tree[first_lvl_feat][first_lvl_val][feat][feat_value][classification] = 0 
                #num occurrences, int to 0
                decider.add(classification) 
                

            second_lvl = (feat, feat_value)
        else: # depth must be 2 (0-based, i.e. level 3)
            first_lvl_feat = first_lvl[0]
            first_lvl_val = first_lvl[1]
            second_lvl_feat = second_lvl[0]
            second_lvl_val = second_lvl[1]

            if not feat in tree[first_lvl_feat][first_lvl_val][second_lvl_feat][second_lvl_val]:
                tree[first_lvl_feat][first_lvl_val][second_lvl_feat][second_lvl_val][feat] = {}
                tree[first_lvl_feat][first_lvl_val][second_lvl_feat][second_lvl_val][feat][feat_value] = {}
            else:
                if not feat_value in tree[first_lvl_feat][first_lvl_val][second_lvl_feat][second_lvl_val][feat]:
                    tree[first_lvl_feat][first_lvl_val][second_lvl_feat][second_lvl_val][feat][feat_value] = {}

            if not classification is None:
                tree[first_lvl_feat][first_lvl_val][second_lvl_feat][second_lvl_val][feat][feat_value][classification] = 0 
               #num occurrences, int to 0
                decider.add(classification) 

    tree["UNMATCHED"] = 0 
    #everything else goes into UNMATCHED besides here
    return (tree, decider)

def tree_test(tree, data_file, decider):
    with open(data_file) as test_data:
         #grab the test data
        data = panda.read_csv(data_file)
        data.columns = [col.strip(' "') for col in data.columns] 
        #for all the col names, just strip the quotes and spaces

    col_name = data.columns
    orig_tree = tree 
    #For each line in the test file, we'll return back to the root of the original tree to traverse down
    for index, row in data.iterrows(): 
        path = []
        for i in range(0, 3): # decision tree can have a max of 3 levels of traversal
            for col in col_name:
                if col in tree: 
                    # Check if we can navigate down the decision tree with this col
                    col_val = row[col].strip(' \'"') 
                    # Get the col's subtree
                    if col_val in tree[col]:
                         #Check to make sure if the subtree is even in the tree or not
                        tree = tree[col][col_val] # It is, so let's iteratively navigate there

        #try to increment the number of occurrences of this test info in the tree to see if we can actually classify it
        find_decision = False
        for decision in decider: 
            #Check if we can possibly make a decision given our data
            if decision in tree:
                tree[decision] += 1
                find_decision = True
                path.append(("decision", decision))
        if not find_decision:
            orig_tree["UNMATCHED"] += 1

        tree = orig_tree 
        #reset our tree so the next test row can start from top

    return orig_tree

def print_tree(tree_text, tree, decider): 
    #once we've trained it and classified everything, print out the occurrences of each class in each subtree
    #this method is similar to the get_tree method 
    # since we have to keep track of where we're at in each level of the tree with first_lvl and second_lvl
    first_lvl = (None, None)
    second_lvl = (None, None)
    for get_info in tree_text:
        depth, feat, feat_value, classification, classification_count = parse_tree(get_info)
        if depth == 0:
            subtree = tree[feat][feat_value]

            find_decision = False
            for decision in decider:
                if decision in subtree:
                    print(feat + " " + str(feat_value) + ": " + decision + " (" + str(subtree[decision]) + ")") 
                    #found decision in tree, print out the counts
                    find_decision = True
            if not find_decision:
                print(feat + " " + str(feat_value))
                 #just found a feature to split data on

            first_lvl = (feat, feat_value)
            second_lvl = (None, None) 
            #reset just in case we went back up the decision tree rather than down
        elif depth == 1:
            first_lvl_feat = first_lvl[0]
            first_lvl_val = first_lvl[1]

            subtree = tree[first_lvl_feat][first_lvl_val][feat][feat_value]

            find_decision = False
            for decision in decider:
                if decision in subtree:
                    print("|   " + feat + " " + str(feat_value) + ": " + decision + " (" + str(subtree[decision]) + ")") 
                    #Found a decision in this tree, print out the counts
                    find_decision = True
            if not find_decision:
                print("|   " + feat + " " + str(feat_value)) 
                #just found a feature to split data on

            second_lvl = (feat, feat_value)
        else: #depth must be 2 (level 3 in the tree since depth is 0-based)
            first_lvl_feat = first_lvl[0]
            first_lvl_val = first_lvl[1]
            second_lvl_feat = second_lvl[0]
            second_lvl_val = second_lvl[1]

            subtree = tree[first_lvl_feat][first_lvl_val][second_lvl_feat][second_lvl_val][feat][feat_value]

            find_decision = False
            for decision in decider:
                if decision in subtree:
                    print("|   |   " + feat + " " + str(feat_value) + ": " + decision + " (" + str(subtree[decision]) + ")")
                     #found a decision in tree, print out the counts
                    find_decision = True
            if not find_decision:
                print("|   |   " + feat + " " + str(feat_value)) 
                #just found a feature to split data on

    unmatched_val = tree["UNMATCHED"]
    if unmatched_val > 0:
         #only print the unmatched results if there's at least 1
        print("UNMATCHED: " + str(tree["UNMATCHED"]))

    return tree_text

if not len(sys.argv) == 3:
    print("Invalid number of arguments!")
    sys.exit()
t_input = sys.argv[1]
test_data = sys.argv[2]
if not os.path.exists(t_input):
    print("Tree file doesn't exist!")
    sys.exit()
elif not os.path.exists(test_data):
    print("Test data doesn't exist!")
    sys.exit()

tree_text = read_input(t_input)
tree_t, decider = get_tree(tree_text)
tree_test = tree_test(tree_t, test_data, decider)
print_tree(tree_text, tree_test, decider)