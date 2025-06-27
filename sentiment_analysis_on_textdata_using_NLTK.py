import nltk
import random
from nltk.corpus import movie_reviews, stopwords
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy

# Download required NLTK datasets
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')

# Feature extractor function
def extract_features(words):
    return {word: True for word in words}

# Load and label the movie reviews dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle the documents for randomness
random.shuffle(documents)

# Prepare feature sets
featuresets = [(extract_features(d),c) for (d, c) in documents]

# Split into training and test sets (80% training, 20% testing)
train_set, test_set = featuresets[:1600], featuresets[1600:]

# Train a Naive Bayes Classifier
classifier = NaiveBayesClassifier.train(train_set)

# Evaluate accuracy
accuracy = nltk_accuracy(classifier, test_set)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Display most informative features
classifier.show_most_informative_features(10)

# Function to analyze sentiment of new input text
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()

def analyze_sentiment(text):
    words = tokenizer.tokenize(text)
    words = [word for word in words if word.lower() not in stopwords.words('english')]
    features = extract_features(words)
    return classifier.classify(features)

# Test the classifier with custom sentences
test_sentences = [
    "This movie is absolutely fantastic! The acting, the story, everything was amazing.",
    "I hated this movie. It was a waste of time and money.",
    "The plot was a bit dull, but the performances were great.",
    "I have mixed feelings about this film. It was okay, not great but not terrible either."
]

for sentence in test_sentences:
    print(f"\nSentence: {sentence}")
    print(f"Predicted Sentiment: {analyze_sentiment(sentence)}")
    print()