# SMS Unsolicited Mail Filter using Naïve Bayes

this is a project of machine learning spam filter that classifies SMS messages as unsolicited mail or proper mail using the Naive Bayes algorithm. this project demonstrates text preprocessing, feature extraction, model training, and classification. this project was made as 2/3 parts of the Johns Hopkins University's specialization on AI and Cybersecurity, specifically Part 2: Advanced Malware and Network Anomaly Detection.

## overview

this spam filter tool processes SMS text messages and determines whether they are spam or legitimate (ham) messages. the model dynamically identifies suspicious keywords and uses machine learning to make accurate classifications.

## features

- **Text Preprocessing**: converts text to lowercase, removes punctuation and stop words
- **Feature Extraction**: uses Bag of Words and TF-IDF approaches
- **Model Training**: trains a Multinomial Naive Bayes classifier
- **Model Evaluation**: provides accuracy, confusion matrix, and classification report
- **Model Export**: saves trained model and vectorizer for future use
- **Classification Function**: allows classification of new messages
- **Interactive Usage**: command-line interface for testing messages

## requirements

- Python 3.x
- scikit-learn
- pandas
- nltk
- joblib

## installation

1. **clone the repository:**
   ```bash
   git clone https://github.com/gbrlprs/naive-bayes-filter/
   cd sms-spam-filter
   ```

2. **install dependencies:**
   ```bash
   pip install nltk scikit-learn pandas joblib
   ```

3. **download NLTK data:**
   ```bash
   python -c "import nltk; nltk.download('stopwords')"
   ```

## usage

### training the Model

Run the main training script to create and export the spam filter model:

```bash
python hopkins.py
```

this will:
- load the SMS dataset (`SMSSpamCollection.csv`)
- preprocess the text data
- extract features using Bag of Words
- train a Naïve Bayes classifier
- evaluate model performance
- export the trained model (`spam_filter_model.pkl`) and vectorizer (`spam_filter_vectorizer.pkl`)

### using the Trained Model

after training, use the model to classify new messages:

```bash
python spam_filter_usage.py
```

this script provides:
- Interactive message classification
- Batch processing capabilities
- Example usage with test messages

### Jupyter Notebook

for educational purposes, use the interactive notebook:

```bash
jupyter notebook SMS_Spam_Filter_Notebook.ipynb
```

## dataset

the project uses the SMS Spam Collection dataset with the following structure:
- **Label**: 'spam' or 'ham'
- **Message**: The SMS text content

## model architecture

### Text Preprocessing Pipeline

1. **Lowercase Conversion**: convert all text to lowercase
2. **Character Filtering**: remove punctuation, numbers, and special characters
3. **Tokenization**: split text into individual words
4. **Stop Word Removal**: remove common English stop words
5. **Text Reconstruction**: join processed tokens back into strings

### Feature Extraction

- **Bag of Words (BoW)**: counts word occurrences in each message
- **TF-IDF**: Term Frequency-Inverse Document Frequency weighting
- **Vectorization**: converts text to numerical feature vectors

### model training

- **Algorithm**: Multinomial Naive Bayes
- **Train/Test Split**: 80% training, 20% testing
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score

## performance

the model typically achieves:
- **Accuracy**: ~95%+
- **Precision**: High precision for spam detection
- **Recall**: Good recall for both spam and ham classification

## project structure

```
sms-spam-filter/
├── hopkins.py                          # Main training script
├── spam_filter_usage.py                # Model usage script
├── SMS_Spam_Filter_Notebook.ipynb      # Interactive notebook
├── SMSSpamCollection.csv               # Dataset file
├── spam_filter_model.pkl               # Trained model (generated)
├── spam_filter_vectorizer.pkl          # Vectorizer (generated)
├── run_spam_filter.bat                 # Windows batch runner
└── README.md                           # This file
```

##  example use

### python script

```python
from spam_filter_usage import load_spam_filter, classify_message

# Load the trained model
model, vectorizer = load_spam_filter()

# Classify a message
message = "WINNER!! You have won a £1000 prize!"
result = classify_message(message, model, vectorizer)
print(f"Classification: {result}")  # Output: Spam
```

### cli

```bash
# Train the model
python hopkins.py

# Use the model interactively
python spam_filter_usage.py
```

## testing

the project includes comprehensive testing with various message types:

- **Spam Examples**: prize notifications, urgent offers, subscription messages
- **Ham Examples**: personal messages, greetings, normal conversations
