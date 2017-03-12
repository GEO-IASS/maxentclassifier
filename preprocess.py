import numpy as np


# Removes all instances that have missing values, and stores remaining
# instances in new .txt file. Was only used once on training data,
# and once on testing data
def process_out_null_values(filename_original, filename_output):
    with open(filename_original) as f:

        with open(filename_output, "w") as w:
            count = 0
            for line in f:
                count +=1

                line_list = line.split(",")
                if not " ?" in line_list:
                    w.write(line)


# Returns a list of strings corresponding to lines of the data file

def getStrings(filename):
    lines = []
    with open(filename) as f:

        for line in f:
            if line != '\n':
                lines.append(line)
    return lines


# Input: a list of the features we will be adding to the feature vector

def createVector(strings, feature_args):
    data = strings.split(", ")
    vec = np.zeros(1)

    # To dynamically generate the feature vectors, we use a set of elif statements to parse the arguments.

    for feature in feature_args:

        # Splits age into 10 different sections. Age ranged from 0-99

        if "age" == feature:
            vec = np.append(vec, np.zeros(10))
            ageFeat = int(data[0].strip()) // 10
            vec[ageFeat] = 1

        elif "workclass" == feature:
            workclasses = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"]
            prev_leng = len(vec)
            vec = np.append(vec, np.zeros(len(workclasses)))
            classFeat = prev_leng + workclasses.index(data[1].strip())
            vec[classFeat] = 1

        elif "education" == feature:
            education = ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"]
            prev_leng = len(vec)
            vec = np.append(vec, np.zeros(len(education)))
            edFeat = prev_leng + education.index(data[3].strip())
            vec[edFeat] = 1

        # Education-num ranged from 1-16

        elif "education-num" == feature:
            prev_leng = len(vec)
            vec = np.append(vec, np.zeros(16))
            ednumFeat = prev_leng + int(data[4].strip()) -1
            vec[ednumFeat] = 1

        elif "marital-status" == feature:
            marital_statuses = ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Separated',
                        'Married-AF-spouse', 'Widowed']
            prev_leng = len(vec)
            vec = np.append(vec, np.zeros(len(marital_statuses)))
            marital_feat = prev_leng + marital_statuses.index(data[5].strip())
            vec[marital_feat] = 1

        elif "occupation" == feature:
            occupations = ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Other-service', 'Sales',
                  'Transport-moving', 'Farming-fishing', 'Machine-op-inspct', 'Tech-support', 'Craft-repair',
                  'Protective-serv', 'Armed-Forces', 'Priv-house-serv']
            prev_leng = len(vec)
            vec = np.append(vec, np.zeros(len(occupations)))
            occupation_feat = prev_leng + occupations.index(data[6].strip())
            vec[occupation_feat] = 1

        elif "capital-gain" == feature:
            prev_leng = len(vec)
            vec = np.append(vec, np.zeros(2))
            # Feature is 1 or 0 depending on whether greater than 3674.
            # This is median value among nonzero capital gains.
            # Overall median of capital gains was just 0
            cap_gain_feat = prev_leng + int(int(data[10].strip()) > 3674)

            vec[cap_gain_feat] = 1

        elif "capital-loss" == feature:
            prev_leng = len(vec)
            vec = np.append(vec, np.zeros(2))
            # Feature is 1 or 0 depending on whether greater than 1876.
            # This is median value among nonzero capital loss.
            # Overall median of capital losses was just 0
            cap_loss_feat = prev_leng + int(int(data[11].strip()) > 1876)
            vec[cap_loss_feat] = 1

        elif "sex" == feature:
            sexes = ["Female", "Male"]
            prev_leng = len(vec)
            vec = np.append(vec, np.zeros(2))
            sex_feat = prev_leng + sexes.index(data[9].strip())
            vec[sex_feat] = 1

        # Splits range of hours-per-week into 10 sections. Hours-per-week in
        # data set ranges from 0-99

        elif "hours-per-week" == feature:
            prev_leng = len(vec)
            vec = np.append(vec, np.zeros(10))
            hpw_feat = prev_leng + int(data[12].strip()) // 10
            vec[hpw_feat] = 1

        elif "race" == feature:
            races = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
            prev_leng = len(vec)
            vec = np.append(vec, np.zeros(len(races)))
            race_feat = prev_leng + races.index(data[8].strip())
            vec[race_feat] = 1

        # Feature encodes whether someone's native country is or is not the US

        elif "native-country" == feature:
            prev_leng = len(vec)
            vec = np.append(vec, np.zeros(2))
            nc_feat = int(data[11].strip() == "United-States")
            nc_feat += prev_leng
            vec[nc_feat] = 1

    label = int(data[-1].strip().rstrip(".") == ">50K")

    return vec, label


# Given a particular feature index, returns a list of all UNIQUE values for that feature in the training data.

def return_Feature_Space(strings, index):
    values = []
    for line in strings:
        features = line.split(",")
        feature = features[index].strip()
        if feature not in values and feature != "0":
            values.append(feature)
    return values


# Given a particular feature index, returns a list of ALL values for that feature in the training data.

def return_all_Features(strings, index):
    values = []
    for line in strings:
        features = line.split(",")
        feature = features[index].strip()
        values.append(feature)
    return values


# Takes a list of strings corresponding to lines from the data file
# returns a list of feature vectors, and a list of correct labels

def create_Feature_Vectors(inputStrings, features):
    vectorList = []
    labelList = []

    for line in inputStrings:
        vec, label = createVector(line, features)
        vectorList.append(vec)
        labelList.append(label)
    return (vectorList, labelList)

# Given filename and desired features, returns a list of feature vectors
# and correct labels


def processData(filename, features):
    strs = getStrings(filename)
    return create_Feature_Vectors(strs, features)


def main():
    # process_out_null_values("testing.txt", "testData.txt")
    # process_out_null_values("adult.data.txt", "processData.txt")
    strs = getStrings("processData.txt")

    cap_gain = return_Feature_Space(strs, 10)

    for i in range(len(cap_gain)):
        cap_gain[i] = int(cap_gain[i])
    print("Average cap_gain = ", sum((cap_gain))/len(cap_gain))
    sort_cap_gain = sorted(cap_gain)
    print("Median Value of cap_gain: ", sort_cap_gain[len(cap_gain)//2])

    cap_loss = return_Feature_Space(strs, 11)
    for i in range(len(cap_loss)):
        cap_loss[i] = int(cap_loss[i])
    print("Average cap_loss = ", sum(cap_loss)/len(cap_loss))
    sort_cap_loss = sorted(cap_loss)
    print("Median Value of cap_gain: ", sort_cap_loss[len(cap_loss)//2])

    print(len(return_Feature_Space(strs, 4)))





if __name__ == "__main__":
    main()
