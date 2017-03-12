# maxEntClassifier.py
# By Jon Bisila and Daniel Lewitz
# Winter 2017 Final Project
# CS 321: Artificial Intelligence

import numpy as np
import preprocess
import warnings
import sys
import os
import itertools
warnings.filterwarnings('error')



def load_data(filename, features):
     return preprocess.processData(filename,features)


def getEmpiricals(instances, labels):

    F = len(instances[0])
    N = len(instances)
    print(F,N)
    empirical1 = np.zeros(F)
    empirical2 = np.zeros(F)
    for i in range(N):
        if labels[i] == 0:
            empirical1 = np.add(empirical1, instances[i])
        else:
            empirical2 = np.add(empirical2, instances[i])
    empirical1 /= N
    empirical2 /= N
    return (empirical1, empirical2)

def getProbs(instance, w0, w1):

    prob0 = np.exp(np.dot(instance, w0))
    prob1 = np.exp(np.dot(instance, w1))
    const = prob0 + prob1
    return(prob0/ const, prob1/const)

def testTraining(instances, labels, weights0, weights1):
    correct = 0
    N = len(instances)
    for i in range(N):
        prob1 = getProbs(instances[i], weights0, weights1)[1]
        guess = int(prob1 > .5)
        correct += int(guess == labels[i])
    return correct/N


def update(instances, weights0, weights1, V, F, N, emp0, emp1):
    model0 = np.zeros(F)
    model1 = np.zeros(F)
    for instance in instances:
        probs = getProbs(instance, weights0, weights1)
        model0 = np.add(model0, instance * probs[0])
        model1 = np.add(model1, instance * probs[1])
    model0 /= N
    model1 /= N
    for i in range(F):
        try:
            weights0[i] = weights0[i] * (emp0[i] / model0[i])**(1/V)
        except:
            weights0[i] = 0
        try:
            weights1[i] = weights1[i] * (emp1[i] / model1[i])**(1/V)
        except:
            weights1[i] = 0

    return weights0, weights1


def maxEnt(features):

    print("MaxEnt Features: ", features)
    instances, labels = load_data("processData.txt", features)
    V = sum(instances[0])
    F = len(instances[0])
    N = len(instances)
    emp0, emp1 = getEmpiricals(instances, labels)
    weights0 = np.ones(F)
    weights1 = np.ones(F)
    testingData, testingLabels = load_data("testData.txt", features)
    #print(testingLabels)

    beforeTesting = testTraining(testingData, testingLabels, weights0, weights1)
    for j in range(20):
        weights0, weights1 = update(instances, weights0, weights1, V, F, N, emp0, emp1)
        # print(testTraining(instances, labels, weights0, weights1))
        # print(weights0)
        # print()
        # print(weights1)
        # print()
    # print(weights0, weights1)
    afterTesting = testTraining(testingData, testingLabels, weights0, weights1)

    return beforeTesting, afterTesting

def compareEachFeature():
    allFeatures = ["age", "workclass", "education", "education-num", "marital-status", "occupation", "capital-gain",
                   "capital-loss", "sex", "hours-per-week"]

    if os.path.isfile("compareIndividualFeatures.txt"):
        os.remove("compareIndividualFeatures.txt")
    with open("compareIndividualFeatures.txt", "a") as f:
        for feature in allFeatures:
            features = [feature]
            before, after = maxEnt(features)
            print("Feature: " + feature, before, after)
            f.write(feature + " " + str(before) + " " + str(after) + "\n")

def comparePairs():
    allFeatures = ["age", "workclass", "education", "education-num", "marital-status", "occupation", "capital-gain",
                   "capital-loss", "sex", "hours-per-week"]
    if os.path.isfile("comparePairs.txt"):
        os.remove("comparePairs.txt")
    with open("comparePairs.txt", "a") as f:
        for i in range(0, len(allFeatures)):
            for j in range(i+1, len(allFeatures)):
                before, after = maxEnt([allFeatures[i], allFeatures[j]])
                f.write("Features " + allFeatures[i] + " & " + allFeatures[j] + ": " + "Before training: " + str(before)
                        + " " + "After training: " + str(after) + "\n")


def compareTriples():
    allFeatures = ["age", "workclass", "education", "education-num", "marital-status", "occupation", "capital-gain",
                   "capital-loss", "sex", "hours-per-week"]
    combos = itertools.combinations(allFeatures, 3)
    if os.path.isfile("compareTriples.txt"):
        os.remove("compareTriples.txt")
    with open("compareTriples.txt", "a") as f:
        for combo in combos:
            before, after = maxEnt(combo)
            f.write("Features " + str(combo) + ": " + "Before training: " + str(before)
                        + " " + "After training: " + str(after) + "\n")

def compareSevens():
    allFeatures = ["age", "workclass", "education", "education-num", "marital-status", "occupation", "capital-gain",
                   "capital-loss", "sex", "hours-per-week"]
    combos = itertools.combinations(allFeatures, 7)
    if os.path.isfile("compareSevens.txt"):
        os.remove("compareSevens.txt")
    with open("compareSevens.txt", "a") as f:
        for combo in combos:
            before, after = maxEnt(combo)
            f.write("Features " + str(combo) + ": " + "Before training: " + str(before)
                        + " " + "After training: " + str(after) + "\n")

def compareEights():
    allFeatures = ["age", "workclass", "education", "education-num", "marital-status", "occupation", "capital-gain",
                   "capital-loss", "sex", "hours-per-week"]
    combos = itertools.combinations(allFeatures, 8)
    if os.path.isfile("compareEights.txt"):
        os.remove("compareEights.txt")
    with open("compareEights.txt", "a") as f:
        for combo in combos:
            before, after = maxEnt(combo)
            f.write("Features " + str(combo) + ": " + "Before training: " + str(before)
                        + " " + "After training: " + str(after) + "\n")

def compareNines():
    allFeatures = ["age", "workclass", "education", "education-num", "marital-status", "occupation", "capital-gain",
                   "capital-loss", "sex", "hours-per-week"]
    combos = itertools.combinations(allFeatures, 9)
    if os.path.isfile("compareNines.txt"):
        os.remove("compareNines.txt")
    with open("compareNines.txt", "a") as f:
        for combo in combos:
            before, after = maxEnt(combo)
            f.write("Features " + str(combo) + ": " + "Before training: " + str(before)
                        + " " + "After training: " + str(after) + "\n")

def main():
    if len(sys.argv) == 1:
        allFeatures = ["age", "workclass", "education", "education-num", "marital-status", "occupation", "capital-gain", "capital-loss"]
        sig_features = ["workclass", "education", "education-num", "capital-gain", "capital-loss", "occupation"]
        # sigbefore, sigafter = maxEnt(sig_features)
        # allbefore, allafter = maxEnt(allFeatures)
        #
        # print("Test Before: ", allbefore, "\n")
        # print("Test After: ", allafter, "\n")
        # print("Training Before: ", otherbefore, "\n")
        # print("Training After: ", otherafter, "\n")
        # print("Significant Features: ", sigbefore, sigafter)
        # print("All Features: ", allbefore, allafter)
        compareEachFeature()
        # comparePairs()
        # compareTriples()
        #compareSevens()
        #compareEights()
        #compareNines()


    else:
        print(maxEnt(sys.argv[1:]))

if __name__ == "__main__":
    main()
