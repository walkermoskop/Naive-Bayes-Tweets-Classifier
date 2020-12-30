import numpy as np
import re
import time
import sys
from sklearn.metrics import accuracy_score
import pickle


class NB(object):
    '''
    A class to build a Naive Bayes model
    ...
    Attributes
    ----------
    laplace: smoother used to prevent zero probabilities for words that don't
                occur for a given class
    word_list: array of all words used in tokenized training data
    class_labels: array of all classes used in training data
    class_priors: array of overall probabilities for each class based on training
                data
    word_freq: matrix of word counts (plus laplaces smoothing) across all classes
    word_given_class: matrix containing P(W|L) for all words and locations (classes)
    class_given_word: matrix containing P(L|W) for all locations and words
    '''
    def __init__(self, laplace=1.0):
        self.laplace = laplace
        self.word_list = None
        self.class_labels = None
        self.class_priors = None
        self.word_freq = None
        self.word_given_class = None
        self.word_prob = None

    def fit(self, X, y):
        '''
        Builds probability matrixes/vectors needed for Naive Bayes and
        returns fitted bayes object
        '''
        ### retreive word list from header row
        self.word_list = X[0]
        ### remove header row from X matrix
        X = np.array(X[1:])
        ### retrieve labels and counts from each class from y array
        self.class_labels, class_counts = np.unique(y_train, return_counts=True)
        ### set as list
        self.class_labels = list(self.class_labels)
        self.class_priors = class_counts / np.sum(class_counts)
        ### build array of word counts for each class (12 total)
        X_class_split = np.array([X[y==label] for label in self.class_labels])
        ### apply laplace smoothing
        self.word_freq = np.array([x.sum(axis=0) for x in X_class_split])\
                            + self.laplace
        self.word_given_class = self.word_freq / self.word_freq.sum(axis=1).reshape(-1,1)
        ### get overall p(word)
        self.word_prob = np.sum(X, axis=0) / np.sum(X)
        self.class_given_word = (self.word_given_class * self.class_priors.reshape(-1,1))/\
                                self.word_prob        
        return self
    
    def predict(self, X):
        '''
        Returns vector containing most likely class for each row in X
        '''
        ### remove header row
        X = np.array(X[1:])
        predicted_classes = []
        for i, xi in enumerate(X):
            ### retrieve largest class probability value for each row in X
            class_idx = np.argmax(np.sum(np.log(self. word_given_class)\
                                    * xi, axis=1) + np.log(self.class_priors.T))
            predicted_classes.append(self.class_labels[class_idx])
        return predicted_classes

            

def tokenize(row):
    '''
    Returns tokenized text and a y array of class outcomes
    Parameters: row of text (an individual tweet)
    '''
    tweet = row.split()
    ### add locations to y vector
    y = tweet.pop(0)
    ### remove punctuation, lowercase, remove non-alphanumeric and stopwords
    tweet = [re.sub(r'\W+', '', t.lower()) for t in tweet]
    tokens = [w for w in tweet if w.isalpha() and w not in stop_words]
    return tokens, y
    

def read_tweets(file, split):
    '''
    For training data, returns raw text and builds up all_words dictionary used
    for word histograms. For test data, simply returns raw text
    '''
    text = open(file).readlines()
    if split=='train':
        for row in text:
            tokens, _ = tokenize(row) ### only care about tokens. ignore the _
            for word in tokens:
                try: all_words[word] +=1
                except KeyError: all_words[word] = 1
    return text


def build_matrix(text): 
    '''
    Returns X matrix (a list of lists) containing tokenized text for all tweets.
    Returns y array (numpy array) containing all class outcomes of tweets
    '''
    ### initialize using all_word_list as header row
    X = [all_word_list]
    y = []
    # ## loop through the tweets again to build out a vector for each tweet
    ### that is the length of all_words_list
    for row in text:
        tokens, yi = tokenize(row)
        array = np.zeros(len(all_word_list))
        ### only include tokens if they're in all_world_list (Otherwise test
        ### matrix will end up with words not seen during training)
        tok_idxs = [all_word_list.index(t) for t in tokens if t in all_word_list]
        for idx in tok_idxs:
            array[idx] +=1
        y.append(yi)
        X.append(array)
    return X, np.array(y)
     

### main program
if __name__== "__main__":
    # start = time.time()

    ### set up global vars
    ### read in stopwords (downloaded from https://www.nltk.org/nltk_data/)
    stop_words = [w.strip('\n') for w in open('nltk_stop_words.txt').readlines()]
    all_words = {}
    
    train_or_test = sys.argv[1]
    
    ## training
    if train_or_test=='train':
        model = sys.argv[2]
        train_input_file = sys.argv[3]
        train_output_file = sys.argv[4]
        
        ## read in text and build all_words dict
        train_text = read_tweets(train_input_file, 'train')
        
        ### convert all words keys to list
        all_words = {key: val for key, val in all_words.items()}
    
        all_word_list = list(all_words.keys())
        # print(len(all_word_list))
        
        ### vectorize tweets
        X_train, y_train = build_matrix(train_text)
        
        if model == 'bayes':        
            bayes = NB(laplace=0.1)
            bayes.fit(X_train, y_train)
            ### write model to external file
            with open(train_output_file, 'wb') as nb_out:
                pickle.dump(bayes, nb_out)
                
            ### for each class, print the top five words for each
            ### first, sort indexes in descending value of probability size
            for i in range(len(np.unique(y_train))):
                prob_idxs = bayes.class_given_word[i].argsort()[::-1]
                top_five = []
                j = 0
                while len(top_five) < 5:
                    ### only add a word if it occurred at least 5 times
                    if bayes.word_freq[i][prob_idxs[j]] >= 5:
                        top_five.append(bayes.word_list[prob_idxs[j]])
                    j+=1
                print('Top 5 words for {}: {}'.format(bayes.class_labels[i],
                      ', '.join([w for w in top_five])))
    

    ### testing
    elif train_or_test=='test': 
        model_input_file = sys.argv[2]
        test_input_file = sys.argv[3]
        test_output_file = sys.argv[4]
        
        test_text = read_tweets(test_input_file, 'test')
        
        ### read in model built during training.
        with open(model_input_file, 'rb') as f:
            clf = pickle.load(f)
            all_word_list = clf.word_list
            
            ### vectorize test tweets
            X_test, y_test = build_matrix(test_text)
            
            test_preds = clf.predict(X_test)
            print(accuracy_score(y_test, test_preds))
            
            ### write results to external file
            with open(test_output_file, 'w') as test_out:
                for i in range(len(y_test)):
                    tweet = ' '.join([t for t in test_text[i].split()[1:]])
                    test_out.writelines('{} {} {}\n'.format(test_preds[i], y_test[i],
                                                     tweet))    
    # print(time.time() - start)
    