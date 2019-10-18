import re
import numpy as np
from collections import Counter
import pandas as pd
# import random
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import download

TRAIN_DATA_PATH = "data/data_train.pkl"
TEST_DATA_PATH = "data/data_test.pkl"

class Naive_Bayes:

    def __init__(self, train_X, train_Y, ):
        self.train_X = train_X
        self.train_Y = train_Y
        self.classes = list(set(train_Y))
        self.stop_words_list = stopwords.words('english')
        self.stemmer = PorterStemmer()
        self.word_stats, self.num_unique_words_in_class, self.total_word_count_per_class = self.get_word_count_per_class()
        self.priors = self.get_priors()

    # pre-processing dataset
    def preprocess_sentence(self, sentence):
        # remove links if needed
        # sentence = re.sub(r'^https?:\/\/.*[\r\n]*', ' website_link ', sentence)
        # remove special characters and stop words
        pattern = re.compile(r'\b\w\w+\b')
        # for word in re.findall(pattern, sentence):
        sentence = [self.stemmer.stem(word) for word in re.findall(pattern, sentence.lower()) if word not in self.stop_words_list]
        
        return sentence

    def get_priors(self):
        priors = {}
        total_examples = len(self.train_Y)
        for class_ in self.classes:
            priors[class_] = np.sum(self.train_Y == class_) / total_examples
        return priors

    # bag of word features - get dictionary of word count per class
    def get_word_count_per_class(self):
        word_stats, num_unique_words_in_class, words_count_per_class = {}, {}, {}
        for class_ in self.classes:
            
            # create a list of all words in the class
            word_list = []
            for review in self.train_X[self.train_Y == class_]:
                word_list += self.preprocess_sentence(review)

            # count the number of unique words and occurrences of each word
            num_unique_words_in_class[class_] = len(set(word_list))
            word_stats[class_] = Counter(word_list)

            # get dictionary of total word count per class
            words_count_per_class[class_] = len(word_list)

        return word_stats, num_unique_words_in_class, words_count_per_class

    # test accuracy
    def accuracy(self, predicted, target):
        return np.mean(predicted == target) * 100.00

    def compute_probability(self, class_, word_list, laplacian, alpha):
        prob = self.priors[class_]
        # override alpha value by setting it 0 if it isn't laplacian
        if not laplacian:
            alpha = 0.0
        denominator_count = self.total_word_count_per_class[class_] + alpha*self.num_unique_words_in_class[class_]
        for word in word_list:
            prob *= ((self.word_stats[class_][word] + alpha) / denominator_count)
        
        return prob
    
    # predict
    def predict_class(self, test_sentences_list, laplacian=True, alpha=0.25):
        # Naive Bayes: for all classes, argmax P(c)*P(d|c)
        predicted_classes = np.zeros(len(test_sentences_list), dtype=np.object)
        for index, test_sentence in enumerate(test_sentences_list):
            word_list = self.preprocess_sentence(test_sentence)
            probabilities = np.zeros(len(self.classes))
            for idx, class_ in enumerate(self.classes):
                probabilities[idx] = self.compute_probability(class_, word_list, laplacian, alpha)

            predicted_classes[index] = self.classes[np.argmax(probabilities)]

        return predicted_classes


# create csv in desired submission format
def create_and_save_submission(predictions, file_name="submission.csv"):
    ids = [i for i in range(len(predictions))]
    sub_df = pd.DataFrame(data=list(zip(ids, predictions)), columns=["Id","Category"])
    sub_df.to_csv(file_name, index=False)

# Return mean model performance after k-fold cross-validation
def evaluate_model(Model, X, y):
    print("Evaluating model performance using k-fold...")
    
    kfold = KFold(
        n_splits=4,
        shuffle=True,
        random_state=42
    )
    
    accuracy = []

    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        my_model = Model(X_train, y_train)
        predictions = my_model.predict_class(X_test)
        accuracy.append(my_model.accuracy(predictions, y_test))
    
    return np.mean(accuracy)

# Train with full dataset; write test data label predictions to .csv file
def predict_test_labels(Model, X, y, X_test):
    my_model = Model(X, y)
    predictions = my_model.predict_class(X_test)
    create_and_save_submission(predictions)

def main():
    print("Downloading stop-words..")
    download('stopwords') # nltk

    train_data = pd.read_pickle(TRAIN_DATA_PATH)
    test_data  = pd.read_pickle(TEST_DATA_PATH)

    X      = np.array(train_data[0])
    y      = np.array(train_data[1])
    X_test = np.array(test_data)

    # Print performance of model after k-fold cross-validation
    accuracy = evaluate_model(Naive_Bayes, X, y)
    print("Accuracy: %.2f"%accuracy)

    # Train with full dataset; write test data label predictions to .csv file
    predict_test_labels(Naive_Bayes, X, y, X_test)

if  __name__ == "__main__":
    main()