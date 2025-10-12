# SMS Spam Filter - Model Usage Script
# This script demonstrates how to use the exported spam filter model

import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download required NLTK data
nltk.download('stopwords')

def preprocess_text(text):
    """
    Clean and preprocess text data by:
    - Converting to lowercase
    - Removing punctuation, special characters, and numbers
    - Removing stop words
    - Tokenizing and rejoining
    """
    stop_words = set(stopwords.words('english'))
    
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

def load_spam_filter():
    """
    Load the trained spam filter model and vectorizer.
    
    Returns:
        tuple: (model, vectorizer) - The loaded model and vectorizer
    """
    try:
        model = joblib.load('spam_filter_model.pkl')
        vectorizer = joblib.load('spam_filter_vectorizer.pkl')
        print("Model and vectorizer loaded successfully!")
        return model, vectorizer
    except FileNotFoundError:
        print("Error: Model files not found. Please run hopkins.py first to train and export the model.")
        return None, None

def classify_message(message, model, vectorizer):
    """
    Classify a new message as spam or ham using the loaded model.
    
    Args:
        message (str): The message to classify
        model: The trained spam filter model
        vectorizer: The fitted vectorizer
        
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

def classify_from_file(file_path, model, vectorizer):
    """
    Classify messages from a text file.
    
    Args:
        file_path (str): Path to the text file containing messages
        model: The trained spam filter model
        vectorizer: The fitted vectorizer
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            messages = file.readlines()
        
        print(f"Classifying {len(messages)} messages from {file_path}:")
        print("-" * 50)
        
        for i, message in enumerate(messages, 1):
            message = message.strip()
            if message:  # Skip empty lines
                classification = classify_message(message, model, vectorizer)
                print(f"Message {i}: {classification}")
                print(f"Text: {message[:100]}{'...' if len(message) > 100 else ''}")
                print("-" * 30)
                
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")

def main():
    """
    Main function to demonstrate spam filter usage.
    """
    print("=== SMS Spam Filter - Model Usage ===\n")
    
    # Load the model and vectorizer
    model, vectorizer = load_spam_filter()
    
    if model is None or vectorizer is None:
        return
    
    # Example 1: Classify individual messages
    print("Example 1: Classifying individual messages")
    print("=" * 50)
    
    test_messages = [
        "Thanks for your subscription to Ringtone UK your mobile will be charged £5/month Please confirm by replying YES or NO.",
        "Hey, how are you doing today? Let's meet for coffee sometime.",
        "WINNER!! As a valued network customer you have been selected to receive £900 prize reward! To claim call 09061701461.",
        "Good morning! Have a great day at work.",
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question."
    ]
    
    for i, message in enumerate(test_messages, 1):
        classification = classify_message(message, model, vectorizer)
        print(f"Message {i}: {classification}")
        print(f"Text: {message}")
        print("-" * 30)
    
    # Example 2: Interactive classification
    print("\nExample 2: Interactive message classification")
    print("=" * 50)
    print("Enter messages to classify (type 'quit' to exit):")
    
    while True:
        user_input = input("\nEnter a message: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if user_input:
            classification = classify_message(user_input, model, vectorizer)
            print(f"Classification: {classification}")
        else:
            print("Please enter a valid message.")

if __name__ == "__main__":
    main()

