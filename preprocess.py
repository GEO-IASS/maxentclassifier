def process_out_null_values():
    with open("adult.data.txt") as f:

        with open("processData.txt", "w") as w:

            for line in f:

                # if " ?" in line.split(","):
                #    print(line)

                if not " ?" in line.split(","):
                    w.write(line + "\n")

                    # else:

                    #    print(line.split(","))

'''
We are splitting data up as follows:

1) For categories like occupation, we will have a feature that is 0/1 for each possible occupation.
2) For categories like age, capital-gain/loss, and hours, we will have a feature that is 0/1 for some range
3) We are making all of the features class dependent on class 1 (less than $50000)
4) We will have all 0's and then a slack value for the feature vector for class 2 (greater than $50000)
'''


def create_Feature_Vectors():
    # TODO: Implement me!