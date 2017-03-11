def process_out_null_values():
    with open("adult.data.txt") as f:

        with open("processData.txt", "w") as w:

            for line in f:

                # if " ?" in line.split(","):
                #    print(line)

                if not " ?" in line.split(","):
                    w.write(line)

                    # else:

                    #    print(line.split(","))

'''
We are splitting data up as follows:

1) For categories like occupation, we will have a feature that is 0/1 for each possible occupation.
2) For categories like age, capital-gain/loss, and hours, we will have a feature that is 0/1 for some range
3) We are making all of the features class dependent on class 1 (less than $50000)
4) We will have all 0's and then a slack value for the feature vector for class 2 (greater than $50000)
'''
import numpy as np
def getStrings(filename):
    lines = []
    with open(filename) as f:

        for line in f:
            if line != '\n':
                lines.append(line)
    #print(lines)
    return lines


def createVector(str):
    vec = np.zeros(34)
    data = str.split(", ")
    ageFeat = int(data[0]) // 10
    vec[ageFeat] = 1
    workclasses = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"]
    classFeat = 10 + workclasses.index(data[1])
    vec[classFeat] = 1
    education = ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"]
    edFeat = 18 + education.index(data[3])
    vec[edFeat] = 1
    label = int(data[-1] == '>50K')

    return (vec, label)


def create_Feature_Vectors(inputStrings):
    vectorList = []
    labelList = []

    for line in inputStrings:
        #print(line)
        vec, label = createVector(line)
        vectorList.append(vec)
        labelList.append(label)
    return (vectorList, labelList)

def processData(filename):
    strs = getStrings(filename)
    return create_Feature_Vectors(strs)

def main():
    #process_out_null_values()
    strs = getStrings("processData.txt")
    print(create_Feature_Vectors(strs))

if __name__ == "__main__":
    main()
