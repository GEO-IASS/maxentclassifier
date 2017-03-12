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
    print(features)
    instances, labels = load_data("processData.txt", features)
    V = sum(instances[0])
    F = len(instances[0])
    N = len(instances)
    emp0, emp1 = getEmpiricals(instances, labels)
    weights0 = np.ones(F)
    weights1 = np.ones(F)
    testingData, testingLabels = load_data("testData.txt", features)

    beforeTesting = testTraining(testingData, testingLabels, weights0, weights1)
    for j in range(20):
        weights0, weights1 = update(instances, weights0, weights1, V, F, N, emp0, emp1)


    afterTesting = testTraining(testingData, testingLabels, weights0, weights1)

    return beforeTesting, afterTesting


def compareN(output, N):
    allFeatures = ["age", "workclass", "education", "education-num", "marital-status", "occupation", "capital-gain",
                   "capital-loss", "sex", "hours-per-week"]
    combos = itertools.combinations(allFeatures, N)
    if os.path.isfile(output):
        os.remove(output)
    with open(output, "a") as f:
        for combo in combos:
            before, after = maxEnt(combo)
            f.write(str(combo) + "," + str(after) + "\n")
def main():
    if len(sys.argv) == 1:
        for i in range(1, 10):
            compareN("compare" + str(i) + ".txt", i)


    else:
        print(maxEnt(sys.argv[1:]))

if __name__ == "__main__":
    main()
