import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pymongo import MongoClient
import json
import urllib.parse
from transformers import RobertaTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # Load credentials from JSON file
    credentials_path = '/Users/khadizatannee/Documents/TRU/Fall-24/IP2/adsc3910-project-group-1/credentials_mongodb.json'

    with open(credentials_path) as f:
        login = json.load(f)

    # Assign credentials to variables
    username = login['username']
    password = urllib.parse.quote(login['password'])  # Ensure the password is URL encoded
    host = login['host']

    # Construct the MongoDB connection string
    url = f"mongodb+srv://{username}:{password}@{host}/?retryWrites=true&w=majority"

    # Connect to MongoDB
    client = MongoClient(url)

    # Select the database you want to use
    db = client['news_database']  # Replace with your database name

    # Drop the collection if it exists to free up space
    try:
        db.drop_collection('news_collection')  # Replace with the collection name you want to delete
        print("Collection dropped successfully.")
    except Exception as e:
        print(f"Error dropping collection: {e}")

    # Select the collection you want to use
    collection = db['news_collection']  # Replace with your collection name

    # Initialize an empty list to store the documents
    documents = []

    # Load the JSON file
    file_path = "../datasets/news_category_dataset_v3.json"
    with open(file_path, 'r') as file:
        for line in file:
            # Each line is a separate JSON object/document
            documents.append(json.loads(line))

    # Number of documents
    num_documents = len(documents)
    print(f"Total number of documents to insert: {num_documents}")

    # Insert the documents into MongoDB
    try:
        collection.insert_many(documents)
        print(f"Inserted {num_documents} documents into MongoDB.")
    except Exception as e:
        print(f"An error occurred while inserting data: {e}")

    # Fetch all documents from the MongoDB collection
    articles = list(collection.find())

    # Convert to a pandas DataFrame
    df = pd.DataFrame(articles)
    df.info()

    # Display the first few rows before cleaning
    print("Data before cleaning:")
    print(df.head())

    # Summary of category column
    df['category'].describe()

    # Check for duplicates in the DataFrame
    has_duplicates = df.duplicated().any()
    print(has_duplicates)

    # Handling missing values
    missing_values = df.isnull().sum()
    print("Missing values in each column:")
    print(missing_values[missing_values > 0])  # Show only columns with missing values

    # Text Cleaning
    def clean_text(text):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
        text = re.sub(r'\[.*?\]', '', text)  # Remove text in square brackets
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
        return text.strip()

    # Apply text cleaning function
    df['cleaned_headline'] = df['headline'].apply(clean_text)
    df['cleaned_short_description'] = df['short_description'].apply(clean_text)

    print(df[['headline', 'cleaned_headline', 'short_description', 'cleaned_short_description']].head())

    # Feature engineering
    df['headline_word_count'] = df['cleaned_headline'].apply(lambda x: len(x.split()))
    df['description_word_count'] = df['cleaned_short_description'].apply(lambda x: len(x.split()))
    df['headline_char_count'] = df['cleaned_headline'].apply(len)
    df['description_char_count'] = df['cleaned_short_description'].apply(len)

    # Tokenization and Encoding
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # Tokenize headline and short_description
    df['headline_tokens'] = df['cleaned_headline'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
    df['description_tokens'] = df['cleaned_short_description'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

    # Label Encoding for Category Column
    label_encoder = LabelEncoder()
    df['category_encoded'] = label_encoder.fit_transform(df['category'])

    ## Calculate and visualize the correlation matrix
    numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.savefig("../results/correlation_matrix.png")
    plt.show()

    # Add text length features
    df['headline_length'] = df['headline'].str.len()
    df['short_description_length'] = df['short_description'].str.len()

    # Visualize the distribution of headline lengths
    plt.figure(figsize=(12, 6))
    sns.histplot(df['headline_length'], bins=30, kde=True)
    plt.title('Distribution of Headline Lengths')
    plt.xlabel('Length of Headline')
    plt.ylabel('Frequency')
    plt.savefig("../results/headline_length_distribution.png")
    plt.show()

    # Visualize the distribution of short description lengths
    plt.figure(figsize=(12, 6))
    sns.histplot(df['short_description_length'], bins=30, kde=True)
    plt.title('Distribution of Short Description Lengths')
    plt.xlabel('Length of Short Description')
    plt.ylabel('Frequency')
    plt.savefig("../results/short_description_lengths_distribution.png")
    plt.show()
    
    df.to_csv("../results/preprocessed_data.csv", index=False)
    print("Preprocessed data saved to preprocessed_data.csv")
    
if __name__ == "__main__":
    main()
    



