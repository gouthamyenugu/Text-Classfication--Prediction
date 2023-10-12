import pickle
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split

# Load the trained model
with open('Toxic_classification_log.pkl', 'rb') as f:
    model = pickle.load(f)

# Load your dataset (assuming it's in a DataFrame format)
# Replace this with your actual dataset loading code
# Example: df = pd.read_csv('your_dataset.csv')
df = pd.read_csv('toxic.csv', sep=',', encoding='latin-1', lineterminator='\n',
                usecols=['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat',
                         'insult', 'identity_hate'])
df.dropna(inplace=True)
df = df[df['toxic'].str.isnumeric()]
cat= ['toxic', 'severe_toxic', 'obscene', 'threat',
       'insult', 'identity_hate']
for i in cat:
    df[i]=df[i].astype(int)
    # Define a function to clean text
    import re
stopwords = ["the", "and", "is", "on", "in", "if", "for", "a", "an", "of", "or", "to", "it", "you", "your"]
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove web links
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove special characters, punctuation marks, and newlines
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra white spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove stopwords
    text = ' '.join(word for word in text.split() if word.lower() not in stopwords)
    return text.lower()

df['comment_text'] = df['comment_text'].apply(clean_text)

X = df['comment_text']
y = df.drop(['id', 'comment_text'], axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=12)

# Create a TfidfVectorizer and fit it on the training documents
vectorizer = TfidfVectorizer()
X_train_tf = vectorizer.fit_transform(X_train)

# Create a TfidfTransformer and transform the training data
tfidf_transformer = TfidfTransformer()
X_train_tfid = tfidf_transformer.fit_transform(X_train_tf)

# Define the XGBoost classifier
logreg_clf = LogisticRegression(random_state=42, class_weight='balanced', tol=0.001, C=1.0)
clf = MultiOutputClassifier(logreg_clf)  # Replace YourClassifier with the desired classifier
model = clf.fit(X_train_tfid, y_train)

# Define a Streamlit function to make predictions
def score_comment(comment):
    # Transform the input comment using the same vectorizer and transformer
    vectorized_comment = vectorizer.transform([comment])
    # Make predictions with the trained model
    results = clf.predict(vectorized_comment)
    text = ''
    for idx, col in enumerate(y.columns):
        # Convert results to float for comparison
        prediction = float(results[0][idx])
        text += '{}: {}\n'.format(col, prediction > 0.5)
    return text

def main():
    st.title("Toxic Comment Classification")
    comment = st.text_input("Enter a comment:")
    # Add a button to trigger predictions
    if st.button("Predict"):
        if comment:
            prediction = score_comment(comment)
            st.text("Predictions:")
            st.text(prediction)

if __name__ == '__main__':
    main()
