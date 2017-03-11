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


# Vector ranges are as follows:
# 0-9: Age Ranges
# 10-17: WorkClasses
# 18-33: Education
# 34-51: Years of Education
# 52-58: Marital Status

def createVector(str):
    vec = np.zeros(73)
    data = str.split(", ")
    ageFeat = int(data[0].strip()) // 10
    vec[ageFeat] = 1
    workclasses = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"]
    classFeat = 10 + workclasses.index(data[1].strip())
    vec[classFeat] = 1
    education = ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"]
    edFeat = 18 + education.index(data[3].strip())
    vec[edFeat] = 1
    ednumFeat = 34 + int(data[4].strip())
    vec[ednumFeat] = 1
    vec[edFeat] = 1

    marital_statuses = ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed']
    marital_feat = 52 + marital_statuses.index(data[5].strip())
    vec[marital_feat] = 1

    job_titles = ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Other-service', 'Sales', 'Transport-moving', 'Farming-fishing', 'Machine-op-inspct', 'Tech-support', 'Craft-repair', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv']
    job_title_feat = 59 + job_titles.index(data[6].strip())
    vec[job_title_feat] = 1

    label = int(data[-1].strip() == ">50K")

    return (vec, label)

def return_Feature_Space(strings, index):
    values = []
    for line in strings:
        features = line.split(",")
        feature = features[index].strip()
        if feature not in values:
            values.append(feature)
    return values




def create_Feature_Vectors(inputStrings):
    vectorList = []
    labelList = []

    for line in inputStrings:
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
    # result = create_Feature_Vectors(strs)
    job_titles = return_Feature_Space(strs, 6)
    print(len(job_titles))


    #Testing that our data outputs as expected...
    # result = create_Feature_Vectors(strs)
    # with open("output.txt", "a") as f:
    #     for item in result:
    #         f.write(str(item))

if __name__ == "__main__":
    main()
