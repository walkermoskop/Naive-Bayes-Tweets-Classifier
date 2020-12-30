# Naive-Bayes-tweets-classifier

The homegrown_naive_bayes.py script was used as part of an assignment for the Indiana University [Elements of AI graduate course](https://luddy.indiana.edu/academics/courses/class/iub-fall-2020-csci-b551#:~:text=CSCI%2DB%20551%20ELEMENTS%20OF%20ARTIFICIAL%20INTELLIGENCE%20(3%20CR.)&text=Principles%20of%20reactive%2C%20goal%2Dbased,%2C%20reasoning%20under%20uncertainty%2C%20planning.). It uses text from tweets to predict the locations of the tweets, and has two primary components:

1) Training a Naive Bayes model using a file of 32K tweets (each tweet contains a location and a tweet).

2) Making location predicitons on an unseen test dataset that contains only tweets.

The script tokenizes and cleans the tweets (stopwords removed, letters are lowercased, special characters and punctuation removed). Words are then vectorized using a simple bag-of-words model, and the ensuing matrices are then passed into the NB class' fit and predict functions.

After training, the model is written to a .pkl file. That model is then used in a separate command line call when making predictions on a test dataset.

To train the model, several command line arguments need to be specified, including the training file name and the name of an output file for the model:

python homegrown_naive_bayes.py train bayes tweets.train.txt bayes-model.pkl

To make predictions on a test set:

python homegrown_naive_bayes.py test bayes-model.pkl tweets.test.txt testing-output-file.txt

These particular train/test files were provided by Indiana University computer science professor [David Crandall](https://homes.luddy.indiana.edu/djcran/).