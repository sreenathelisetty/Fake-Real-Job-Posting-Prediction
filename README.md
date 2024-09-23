# Fake/Real Job Posting Prediction

## Objective
The goal of this project is to identify and classify job postings as either legitimate or fraudulent using machine learning techniques. This project aims to provide an automated solution to detect fake job postings, enhancing user trust on job platforms.

## Dataset
- **Source:** [Kaggle Dataset - Real or Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction/data)
- **Number of Records:** 17,880 job postings
- **Number of Attributes:** 18
  - Attributes include: `job_id`, `title`, `location`, `company_profile`, `description`, `requirements`, `benefits`, `fraudulent`, etc.
- **Target Variable:** `fraudulent` (binary classification: 0 = real, 1 = fake)

## Data Preprocessing
Steps taken to clean and preprocess the dataset:
1. **Missing Values Handling:** Removed or imputed missing data where applicable.
2. **Text Preprocessing:**
   - Converted text to lowercase.
   - Tokenized text using `nltk`.
   - Removed stopwords and non-informative characters.
   - Used `TfidfVectorizer` to convert job descriptions to numerical form.
3. **Feature Selection:** Selected the most relevant features (`description`, `company_profile`, `requirements`, etc.) for training.

```python
# Sample code for text preprocessing
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Tokenization and stopword removal
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Example function to preprocess text
def preprocess_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]  # Remove non-alphabetical tokens
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return ' '.join(tokens)

# Example: Preprocessing job descriptions
data['cleaned_description'] = data['description'].apply(preprocess_text)
Model Training

We used classification algorithms to predict whether a job posting is fake or real. The key models we applied include:

Logistic Regression
Random Forest
Support Vector Machine (SVM)
Decision Tree (tuned with GridSearchCV)
Splitting the Data:
python
Copy code
from sklearn.model_selection import train_test_split

# Splitting data into training and testing sets
X = data['cleaned_description']
y = data['fraudulent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
Model Implementation:
python
Copy code
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Training a Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Predicting on test set
y_pred = rf_model.predict(X_test)

# Model evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
Results

The Random Forest model achieved high accuracy in identifying both real and fake job postings. The precision for fraudulent job postings improved significantly after model tuning, with an accuracy rate of 86%.

Conclusion

This model can serve as an efficient tool for job platforms to automatically detect fraudulent postings and protect users from scams. In future work, we plan to explore additional NLP techniques and alternative classification models to improve the accuracy further.
