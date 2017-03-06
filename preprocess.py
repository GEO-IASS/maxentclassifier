with open("adult.data.txt") as f:

    with open("processData.txt", "w") as w:

        for line in f:

            #if " ?" in line.split(","):
            #    print(line)

            if not " ?" in line.split(","):

                w.write(line + "\n")

            #else:

            #    print(line.split(","))
