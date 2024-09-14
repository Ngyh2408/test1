import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Download NLTK resources
nltk.download('stopwords')

import pandas as pd
import zipfile

# Define the path to the ZIP file
zip_file_path = 'ass.zip'  # Path to the ZIP file
csv_file_name = 'dataset.csv'  # Name of the CSV file inside the ZIP

# Extract the CSV file from the ZIP archive and read it
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    with zip_ref.open(csv_file_name) as file:
        data = pd.read_csv(file)

# Display the first few rows to inspect the data
print(data.head())


# Assuming 'review' is the column containing the product reviews
# and 'label' (optional) is the column with sentiment (1 for positive, 0 for negative)
# Adjust these column names as needed based on your dataset's structure
reviews = data['review']
# If your dataset doesn't have labels, you'll need to manually label the reviews for training purposes.
# labels = data['label']  

# Preprocessing function
def preprocess_text(text):
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    # Lowercase the text
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # Apply stemming
    stemmer = PorterStemmer()
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

# Apply preprocessing to the reviews
data['cleaned_review'] = reviews.apply(preprocess_text)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(data['cleaned_review'], data['label'], test_size=0.2, random_state=42)

# Create a text classification pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('classifier', MultinomialNB())
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy}")
print("Classification Report:")
print(report)
