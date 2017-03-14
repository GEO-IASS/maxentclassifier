IMPORTANT: This code was written in Python 3. Please run it with the Python3 shell.

To see a list of all possible run options: run "python maxEntClassifier.py --help"

If you are having trouble with, please continue reading. To read about our files, please skip to the "Notes" section.

Currently, our classifier has a few possible options. If you just run it with no flag or additional parameters,
it will train a classifier using every feature separately and then run it on the testing data to output a result.

To list all possible features, run "maxEntClassifier.py --features"

If you run it with ""-w" as the last parameter, it will print out the weights as it runs iterations.

If you want to run it with particular features, run "maxEntClassifier.py " followed by each feature split by a space.

To run with joint features, just include a "+" between the two features. Please read the note before for more info.

If you run "maxEntClassifier.py -cs", it will compare a set of individual features to the set of joint features
generated from those individual features.

If you run "maxEntClassifier.py -c", it will compare all possible combinations of all non-joint features.
WARNING: This will take a REALLY long time. Don't run this unless you don't want to use your computer for another
         three days.


                                        NOTES:

1) No auxiliary data files or modules should be needed. processData.txt and testData.txt are
   the training and testing data that the program uses. These are edited to not include instances
   with missing entries. adult.data.txt and testing.txt are the original training and testing files.
   The other files are for our data analysis and should be ignored.

2) If joint features currently only include workclass, sex, education, marital-status, occupation, and race so if
   you try to run it with other features, it will do it's best but inevitably not give you what you want.



