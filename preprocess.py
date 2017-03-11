def process_out_null_values(filename_original, filename_output):
    with open(filename_original) as f:

        with open(filename_output, "w") as w:
            count = 0
            for line in f:
                count +=1

                # if " ?" in line.split(","):
                #    print(line)

                if not " ?" in line.split(","):
                    w.write(line)

                    # else:

                    #    print(line.split(","))
            print(count)

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
# 59-73: Job Title
# 74-83: Hours-per-week (removed)
# 84-85: Sex (removed)
# 74-75: Capital Gain < or > 5000
# 76-77: Capital Loss < or > 1750

def createVector(str):
    vec = np.zeros(78)
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

    cap_gain_feat = 74 + int(int(data[10].strip()) > 5000)
    vec[cap_gain_feat] = 1

    cap_loss_feat = 76 + int(int(data[11].strip()) > 1750)
    vec[cap_loss_feat] = 1


    # These two features actually reduced classification probability so I removed them.

    # hourspw_feat = 74 + int(data[12].strip()) // 10
    # vec[hourspw_feat] = 1
    #
    # sexes = ["Female", "Male"]
    # sex_feat = 84 + sexes.index(data[9].strip())
    # vec[sex_feat] = 1

    label = int(data[-1].strip() == ">50K.")

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
    process_out_null_values("testing.txt", "testData.txt")
    strs = getStrings("processData.txt")
    # result = create_Feature_Vectors(strs)
    capital_loss = return_Feature_Space(strs, 11)
    print(sorted(capital_loss))


    greater_3500 = 0
    for i in range(len(capital_loss)):
        if int(capital_loss[i]) > 1900:
            greater_3500 += 1
        capital_loss[i] = int(capital_loss[i])
    print(greater_3500)

    print("Average: ", sum(capital_loss)/len(capital_loss))
    print(len(capital_loss))



    #Testing that our data outputs as expected...
    # result = create_Feature_Vectors(strs)
    # with open("output.txt", "a") as f:
    #     for item in result:
    #         f.write(str(item))

if __name__ == "__main__":
    main()
