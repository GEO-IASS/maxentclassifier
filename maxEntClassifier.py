# maxEntClassifier.py
# By Jon Bisila and Daniel Lewitz
# Winter 2017 Final Project
# CS 321: Artificial Intelligence

import numpy as np
import preprocess
import warnings
warnings.filterwarnings('error')



def load_data(filename):
     return preprocess.processData(filename)


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
        guess = int(prob1 >= .5)
        correct += (guess == labels[i])
    print(correct/N)



def maxEnt():

    def update():
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
        #weights0[0] = 0
        #weights1[0] = 0

    instances, labels = load_data("processData.txt")
    V = sum(instances[0])
    F = len(instances[0])
    N = len(instances)
    emp0, emp1 = getEmpiricals(instances, labels)
    weights0 = np.ones(F)
    weights1 = np.ones(F)
    testingData, testingLabels = load_data("testData.txt")
    testTraining(testingData, testingLabels, weights0, weights1)
    for j in range(20):
        update()
        # print(testTraining(instances, labels, weights0, weights1))
        # print(weights0)
        # print()
        # print(weights1)
        # print()
    testTraining(testingData, testingLabels, weights0, weights1)






def classifyingProb(feature_vector, V, weights):
    prob_class_one = (np.exp(np.dot(weights, feature_vector)))
    prob_class_two = (np.exp(weights[-1] * V))
    normalize_constant = prob_class_one + prob_class_two

    prob_class_one = prob_class_one/normalize_constant
    prob_class_two = prob_class_two/normalize_constant

    return (prob_class_one, prob_class_two)

def main():
    maxEnt()

if __name__ == "__main__":
    main()
