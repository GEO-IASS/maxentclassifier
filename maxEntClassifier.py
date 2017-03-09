# maxEntClassifier.py
# By Jon Bisila and Daniel Lewitz
# Winter 2017 Final Project
# CS 321: Artificial Intelligence

import numpy as np



def load_data(filename):
     return preprocess.processData(filename)

def maxEnt():
    data = load_data(filename())
    length = len(data[[0]-1)
    V = 13
    slack_vector = np.zeros(length + 1)
    slack_vector[-1] = V


    # First, Calculate empirical distributions.
    empiricals = np.zeros(length+1)
    for trainingInstance in data:
        # if class is 1:
        if trainingInstance[-1] = 1:
            for i in range(length):
                empiricals[i] += trainingInstance[i]

        else:
            empiricals[-1] += V

    empiricals = empiricals * 1.0/len(data)

    weights = np.ones(len(empiricals))

    while not_converged():
        E_p = 0
        for instance in data:
            E_p += classifyingProb(instance, V, weights)[0] *




def classifyingProb(feature_vector, V, weights):
    prob_class_one = (np.exp(np.dot(weights, feature_vector)))
    prob_class_two = (np.exp(weights[-1] * V))
    normalize_constant = prob_class_one + prob_class_two

    prob_class_one = prob_class_one/normalize_constant
    prob_class_two = prob_class_two/normalize_constant

    return (prob_class_one, prob_class_two)







