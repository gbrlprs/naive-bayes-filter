# SMS Spam Collection using Naïve Bayes Algorithm
# Johns Hopkins University - Hands-on Lab

# Step 1: Import Required Packages
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib  # For model export/import

# Download required NLTK data
nltk.download('stopwords')

# Step 2: Load and Preprocess Data
print("Loading dataset...")
data = pd.read_csv('SMSSpamCollection.csv')  # Replace with your dataset path
print("Dataset loaded successfully!")
print(f"Dataset shape: {data.shape}")
print("\nFirst 5 rows:")
print(data.head())

# Convert labels to binary values (1 for spam, 0 for ham)
data['Label'] = data['Label'].map({'spam': 1, 'ham': 0})
print(f"\nLabel distribution:")
print(data['Label'].value_counts())

# Step 3: Text Preprocessing
print("\nPreprocessing text data...")
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Clean and preprocess text data by:
    - Converting to lowercase
    - Removing punctuation, special characters, and numbers
    - Removing stop words
    - Tokenizing and rejoining
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation, special characters, and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize the text (split into words)
    tokens = text.split()
    # Remove stop words
    tokens = [word for word in tokens if word not in stop_words]
    # Join the tokens back into a single string
    return ' '.join(tokens)

# Apply preprocessing to the 'Message' column
data['cleaned_message'] = data['Message'].apply(preprocess_text)

# Display the first few rows of the cleaned data
print("\nCleaned data preview:")
print(data[['Message', 'cleaned_message']].head())

# Step 4: Feature Extraction
print("\nExtracting features...")

# Bag of Words (BoW) approach
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(data['cleaned_message'])

# Convert the BoW matrix to a DataFrame for better readability (optional)
bow_df = pd.DataFrame(X_bow.toarray(), columns=vectorizer.get_feature_names_out())
print(f"Bag of Words shape: {X_bow.shape}")
print("BoW sample:")
print(bow_df.head())

# TF-IDF approach (alternative)
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(data['cleaned_message'])

# Convert the TF-IDF matrix to a DataFrame for better readability (optional)
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
print(f"\nTF-IDF shape: {X_tfidf.shape}")
print("TF-IDF sample:")
print(tfidf_df.head())

# Step 5: Prepare Training Data
print("\nPreparing training data...")

# Define the feature matrix and target vector
X = data['cleaned_message']  # Feature matrix (cleaned messages)
y = data['Label']  # Target vector (labels: 1 for spam, 0 for ham)

# For Bag of Words (using CountVectorizer):
X_features = vectorizer.fit_transform(X)

# For TF-IDF (uncomment the following line if you prefer TF-IDF):
# X_features = tfidf_vectorizer.fit_transform(X)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

# Display the size of each set
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Step 6: Train the Model
print("\nTraining Naïve Bayes model...")
model = MultinomialNB()
model.fit(X_train, y_train)
print("Model training completed!")

# Step 7: Evaluate the Model
print("\nEvaluating model performance...")

# Predict the labels for the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Generate and print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Generate and print the classification report
class_report = classification_report(y_test, y_pred, target_names=['ham', 'spam'])
print("\nClassification Report:")
print(class_report)

# Step 8: Create Classification Function
def classify_message(message):
    """
    Classify a new message as spam or ham using the trained model.
    
    Args:
        message (str): The message to classify
        
    Returns:
        str: "Spam" or "Ham"
    """
    # Preprocess the message
    cleaned_message = preprocess_text(message)
    # Transform using the same vectorizer used for training
    vec_message = vectorizer.transform([cleaned_message])
    # Make prediction
    prediction = model.predict(vec_message)
    return "Spam" if prediction[0] == 1 else "Ham"

# Step 9: Test the Model
print("\nTesting the model with sample messages...")

# Example usage
new_message = "Thanks for your subscription to Ringtone UK your mobile will be charged £5/month Please confirm by replying YES or NO. If you reply NO you will not be charged"
classification = classify_message(new_message)
print(f"Message: {new_message}")
print(f"Classification: {classification}")

# Test with a ham message
ham_message = "Hey, how are you doing today? Let's meet for coffee sometime."
ham_classification = classify_message(ham_message)
print(f"\nMessage: {ham_message}")
print(f"Classification: {ham_classification}")

# Step 10: Export the Model
print("\nExporting the trained model...")
joblib.dump(model, 'spam_filter_model.pkl')
joblib.dump(vectorizer, 'spam_filter_vectorizer.pkl')
print("Model and vectorizer exported successfully!")

print("\n=== SMS Spam Filter Training Complete ===")
print("The model has been trained and exported. You can now use it to classify new messages.")