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
import zipfile
import os

# Download NLTK resources
nltk.download('stopwords')

# Define the path to the ZIP file
zip_file_path = 'ass.zip'  # Change if needed
csv_file_name = 'Dataset-SA.csv'  # Change if needed

# Check if the ZIP file exists
if not os.path.exists(zip_file_path):
    print(f"Error: The ZIP file '{zip_file_path}' does not exist.")
else:
    # Extract the CSV file from the ZIP archive and read it
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # List the contents of the ZIP file
        print("Files in ZIP archive:", zip_ref.namelist())
        
        # Check if the specified CSV file exists in the ZIP
        if csv_file_name not in zip_ref.namelist():
            print(f"Error: The file '{csv_file_name}' is not found in the ZIP archive.")
        else:
            with zip_ref.open(csv_file_name) as file:
                data = pd.read_csv(file)

            # Display the first few rows and column names
            print("First few rows of the dataset:")
            print(data.head())
            print("\nColumn names in the dataset:", data.columns)

            # Attempt to locate the 'review' and 'label' columns
            if 'review' not in data.columns:
                print("Error: Column 'review' not found in the dataset.")
            else:
                reviews = data['review']

                # Check for the label column, if it exists
                if 'label' in data.columns:
                    labels = data['label']
                else:
                    # Handle the case where there are no labels
                    print("Warning: Column 'label' not found. Generating random labels for demonstration purposes.")
                    labels = np.random.randint(0, 2, size=len(reviews))

                # Preprocessing function
                def preprocess_text(text):
                    # Remove non-alphabetic characters
                    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
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
                X_train, X_test, y_train, y_test = train_test_split(data['cleaned_review'], labels, test_size=0.2, random_state=42)

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

                print(f"\nModel Accuracy: {accuracy}")
                print("\nClassification Report:")
                print(report)
