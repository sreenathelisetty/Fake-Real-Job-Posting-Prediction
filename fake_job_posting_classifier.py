
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load your dataset here (as an example)
import pandas as pd

# Load dataset (replace with actual dataset path)
data = pd.read_csv('fake_job_postings.csv')

# Data Preprocessing
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]  # Remove non-alphabetical tokens
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return ' '.join(tokens)

# Apply preprocessing to the 'description' column
data['cleaned_description'] = data['description'].apply(preprocess_text)

# Splitting the dataset into training and testing sets
X = data['cleaned_description']
y = data['fraudulent']  # Target column indicating if a job posting is fake or real
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Vectorize text data using TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model Training
rf_model = RandomForestClassifier()
rf_model.fit(X_train_tfidf, y_train)

# Model Prediction
y_pred = rf_model.predict(X_test_tfidf)

# Model Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model for future use (optional)
import joblib
joblib.dump(rf_model, 'fake_job_classifier_model.pkl')
