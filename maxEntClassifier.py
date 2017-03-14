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


# Creates empiricals given the set of feature vectors, and the true labels
def getEmpiricals(instances, labels):

    F = len(instances[0])
    N = len(instances)
    # Empirical1[i] will store the probability that some instance will have true
    # label 0, and will have a 1 for feature i
    empirical0 = np.zeros(F)
    empirical1 = np.zeros(F)
    for i in range(N):
        # If the true label is 0, adds features from instance to empirical0
        if labels[i] == 0:
            empirical0 = np.add(empirical0, instances[i])
        else:
            empirical1 = np.add(empirical1, instances[i])

    # Normalizes empiricals
    empirical0 /= N
    empirical1 /= N
    return empirical0, empirical1


# Given a feature vector, and current weight vectors, returns probability
# that instance is from class 0 or class 1
def getProbs(instance, w0, w1):
    # print("w0 :", w0, " w1: ", w1)
    # print("\n")

    prob0 = np.exp(np.dot(instance, w0))
    prob1 = np.exp(np.dot(instance, w1))
    const = prob0 + prob1
    return prob0 / const, prob1 / const


# Classifies feature vectors from instances, and returns the proportion of
# guesses that are correct
def testTraining(instances, labels, weights0, weights1):
    correct = 0
    error = 0
    N = len(instances)
    for i in range(N):
        prob1 = getProbs(instances[i], weights0, weights1)[1]
        guess = int(prob1 > .5)
        correct += int(guess == labels[i])
        error += abs(prob1 - labels[i])
    return correct/N, error/N


# Performs a full iteration of maxent, and updates the weights
def update(instances, weights0, weights1, V, F, N, emp0, emp1):

    # Model0, model1 are the feature distributions as per our model given the
    # current weights

    model0 = np.zeros(F)
    model1 = np.zeros(F)
    for instance in instances:
        probs = getProbs(instance, weights0, weights1)
        model0 = np.add(model0, instance * probs[0])
        model1 = np.add(model1, instance * probs[1])
    model0 /= N
    model1 /= N
    for i in range(F):

        # If the model prediction is 0 for some feature, sets
        # corresponding weight to 0. In some cases, this may be due
        # to python rounding down very small numbers to 0
        try:
            weights0[i] *= (emp0[i] / model0[i]) ** (1 / V)
        except:
            weights0[i] = 0
        try:
            weights1[i] *=(emp1[i] / model1[i]) ** (1 / V)
        except:
            weights1[i] = 0

    return weights0, weights1


# Creates and tests a maximum entropy model, given a list of desired features
def maxEnt(features, withWeights = False):
    print("Running maxent with features: " , features)
    instances, labels = load_data("processData.txt", features)
    V = sum(instances[0])
    F = len(instances[0])
    N = len(instances)
    emp0, emp1 = getEmpiricals(instances, labels)
    weights0 = np.ones(F)
    weights1 = np.ones(F)
    changes0 = np.ones(F)
    changes1 = np.ones(F)
    testingData, testingLabels = load_data("testData.txt", features)

    beforeTestingCorrect, beforeTestingError = testTraining(testingData, testingLabels, weights0, weights1)

    updateNum =0
    #Continues updating weights while average change of weight entry
    #is > 10^(-6)
    while max(abs(sum(changes0)), abs(sum(changes1))) > .000001 * F and updateNum <= 30:
        oldWeights0, oldWeights1 = np.copy(weights0), np.copy(weights1)
        weights0, weights1= update(instances, weights0, weights1, V, F, N, emp0, emp1)
        changes0 = weights0 - oldWeights0
        changes1 = weights1 - oldWeights1

        if withWeights:

            print("Update" , updateNum ,"\n \n ","w0:", weights0, "\n \n", "w1:" , weights1, "\n \n ")
        updateNum +=1


    afterTestingCorrect, afterTestingError = testTraining(testingData, testingLabels, weights0, weights1)
    return afterTestingCorrect, afterTestingError


# Runs maxent on all possible combinations of N features, and writes results to
# the desired output file
def compareN(output, N):
    allFeatures = ["age", "workclass", "education", "education-num", "marital-status", "occupation", "capital-gain",
                   "capital-loss", "race", "native-country", "hours-per-week", "sex"]
    combos = itertools.combinations(allFeatures, N)
    if os.path.isfile(output):
        os.remove(output)
    with open(output, "a") as f:
        for combo in combos:
            afterCorrect, afterError = maxEnt(combo)
            f.write(str(combo).replace(",", " ") + "," + str(afterCorrect) + "," + str(afterError) + "\n")

def compareJointToSingles(combos):
    if os.path.isfile("combosVsSingles.csv"):
        os.remove("combosVsSingles.csv")
    with open("combosVsSingles.csv", "a") as w:
        for combo in combos:
            comboToAppend = str(combo[0]) + "+" + str(combo[1])
            comboCorrect, comboError = maxEnt([comboToAppend])
            singles = [combo[0], combo[1]]

            singlesCorrect, singlesError = maxEnt(singles)

            w.write("Combo" + "," + str(comboToAppend) + "," + str(comboCorrect) + "," + str(comboError) + "\n")
            w.write("Singles" + "," + str(singles).replace(",", " ") + "," + str(singlesCorrect) + "," + str(singlesError) + "\n")


#Displays commandline options
def getHelp():
        print("\n")
        print("Usage:")
        print("  maxEntClassifier.py [features] [options]\n")
        print("Options: \n")
        print("  maxEntClassifier.py [--features] \t Lists all possible features. \n")
        print("  maxEntClassifier.py \t \t \t Classifies using all features. \n")
        print("  maxEntClassifier.py [features]  \t Classifies using specified features. \n")
        print("  maxEntClassifier.py [features] [-w] \t Displays weights while running. \n")
        print("  maxEntClassifier.py [-c] \t \t Classifies using every possible combination of non-joint features.\n")
        print("  maxEntClassifier.py [-cs] \t \t Compares set of individual features to set of joint features.\n")


# Runs maxEnt with the desired features
def main():
    if len(sys.argv) > 1:

        if sys.argv[1] == "--help":
            getHelp()
            sys.exit()

        elif sys.argv[-1] == "--features":
            allFeatures = "Possible Features include: age, workclass, education, education-num, marital-status, occupation, capital-gain, capital-loss, race, native-country, hours-per-week, and sex. \n"
            print("\n")
            print(allFeatures)
            print("Currently, only workclass, sex, education, marital-status, occupation, and race are available as joint-features.\n")

        elif "-w" == sys.argv[-1] and len(sys.argv) > 2:
            afterCorrect, afterError = maxEnt(sys.argv[1:-1], withWeights=True)
            print("After Testing Correct: ", afterCorrect, "\t", "After Testing Error: ", afterError)

        elif "-w" == sys.argv[-1] and len(sys.argv) == 2:
            allFeatures = ["age", "workclass", "education", "education-num", "marital-status", "occupation",
                           "capital-gain", "capital-loss", "race", "native-country", "hours-per-week", "sex"]
            afterCorrect, afterError = maxEnt(allFeatures, withWeights=True)
            print("After Testing Correct: ", afterCorrect, "\t", "After Testing Error: ", afterError)

        elif "-c" == sys.argv[-1]:
            allFeatures = ["age", "workclass", "education", "education-num", "marital-status", "occupation",
                           "capital-gain", "capital-loss", "race", "native-country", "hours-per-week", "sex"]
            for i in range(1, len(allFeatures)):
                compareN("compare" + str(i) + ".csv", i)

        elif "-cs" == sys.argv[-1]:
            jointable_features = ["workclass", "education", "marital-status", "occupation", "sex", "race"]

            jointable_combos = list(itertools.combinations(jointable_features, 2))
            combos = []
            for combo in jointable_combos:
                comboToAppend = str(combo[0]) + "+" + str(combo[1])
                combos.append(comboToAppend)
            afterCorrect, afterError = maxEnt(jointable_features)
            jointCorrect, jointError = maxEnt(combos)
            print("After allFeatures Correct: ", afterCorrect, "\t", "After allFeatures Error: ", afterError)
            print("After jointFeatures Correct: ", jointCorrect, "\t", "After jointFeatures Error: ", jointError)


        else:
            afterCorrect, afterError = maxEnt(sys.argv[1:])
            print("After Testing Correct: ", afterCorrect, "\t", "After Testing Error: ", afterError)

    else:
        allFeatures = ["age", "workclass", "education", "education-num", "marital-status", "occupation", "capital-gain",
                       "capital-loss", "race", "native-country", "hours-per-week", "sex"]
        # jointable_features = ["workclass", "education", "marital-status", "occupation", "sex", "race"]

        # jointable_combos = list(itertools.combinations(jointable_features, 2))

        # compareJointToSingles(jointable_combos)
        # for combo in jointable_combos:
        #     comboToAppend = str(combo[0]) + "+" + str(combo[1])
        #     allFeatures.append(comboToAppend)
        afterCorrect, afterError = maxEnt(allFeatures)
        print("After Testing Correct: ", afterCorrect, "\t", "After Testing Error: ", afterError)


if __name__ == "__main__":
    main()
