from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Read the dataset (trainset.txt) file
def read_corpus(corpus_file, use_sentiment):
    documents = []  
    labels = []  
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()  
            documents.append(tokens[3:]) 

            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append(tokens[1])
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append(tokens[0])

    return documents, labels


# a dummy function that just returns its input
def identity(x):
    return x

def display(test_class, test_guess, classifier):
    print("\n# Accuracy:",accuracy_score(test_class, test_guess))  

    print("\n# Classification Report:")
    print(classification_report(test_class, test_guess))  

    print("# Confusion Matrix:")
    cm = confusion_matrix(test_class, test_guess, labels=classifier.classes_)  
    print(classifier.classes_) 
    print(cm) 
    print("________________________________________________________\n")

def Naive_Bayes(sentiment, tfidf):
    # read full data set and separate document and class
    X, Y = read_corpus('trainset.txt', use_sentiment=sentiment) 
    split_point = int(0.75 * len(X)) 

    train_doc = X[:split_point] 
    train_class = Y[:split_point] 

    test_doc = X[split_point:] 
    test_class = Y[split_point:] 

    # we use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    if tfidf: # for feature vectorizer
        vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
    else: # for count vectorizer
        vec = CountVectorizer(preprocessor=identity, tokenizer=identity)

    # combine vectorizer with Naive Bayes classifier
    classifier = Pipeline([('vec', vec), ('cls', MultinomialNB())])


    # build model by train_doc and classifier
    classifier.fit(train_doc, train_class)

    # store label in test_guess
    test_guess = classifier.predict(test_doc)


    # Print the results
    display(test_class, test_guess, classifier)


if __name__ == "__main__":

    print("\n<<===== Topic Analysis ( Count Vectorizer ) =====>>")
    Naive_Bayes(False, False) 

    print("\n<<===== Topic Analysis ( Feature Vectorizer ) =====>>")
    Naive_Bayes(False, True) 

    print("\n<<===== Sentiment Analysis ( Count Vectorizer ) =====>>")
    Naive_Bayes(True, False) 

    print("\n<<===== Sentiment Analysis ( Feature Vectorizer ) =====>>")
    Naive_Bayes(True, True)