# -*- coding: utf-8 -*-

import pandas as pd

# Load the dataset
file_path = 'fake_job_postings.csv'
df = pd.read_csv(file_path)
df.head()

# Get descriptive statistics
desc_stats = df.describe()

# Get the number of rows and columns
num_rows, num_columns = df.shape

# Get the name of the columns
column_names = df.columns

# Get the type of the columns
column_types = df.dtypes

# Display the information using print statements
print("This is the descriptive statistics:")
desc_stats

print(f"\nThe dataset has {num_rows} rows and {num_columns} columns.")

print("\nThe names of the columns are:")
print(column_names)

print("\nThe types of the columns are:")
print(column_types)

# Using the info() method to get a concise summary of the dataframe
print("\nInformation about the DataFrame:")
df_info = df.info()

# Calculate the number of null values in each column
df.isnull().sum()

# Identify categorical columns (object type columns in this case)
categorical_columns = df.select_dtypes(include=['object']).columns

# Calculate the value counts for each categorical column
value_counts = {column: df[column].value_counts() for column in categorical_columns}

value_counts

# Extracting the categorical and the numerical values
categorical=[]
numerical=[]

for col in df.columns:
    if df[col].dtypes !='object':
            numerical.append(col)
    else:
        categorical.append(col)

print(len(categorical))
print(categorical)

import seaborn as sns
sns.heatmap(df[numerical].corr(), cmap='coolwarm', annot=True)

"""# What is the distribution of fraudulent versus non-fraudulent job postings?"""

import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# 1. Distribution of fraudulent vs non-fraudulent job postings
plt.figure(figsize=(8, 6))
sns.countplot(x='fraudulent', data=df)
plt.title('Distribution of Fraudulent vs Non-Fraudulent Job Postings')
plt.xlabel('Fraudulent Job Posting')
plt.ylabel('Count')
plt.show()

"""# How does the presence of a company logo relate to job posting authenticity?"""

# 2. Presence of a company logo and job posting authenticity
plt.figure(figsize=(8, 6))
sns.countplot(x='fraudulent', hue='has_company_logo', data=df)
plt.title('Company Logo Presence and Job Posting Authenticity')
plt.xlabel('Fraudulent Job Posting')
plt.ylabel('Count')
plt.show()

"""# What are the top industries with the highest number of job postings?"""

# 3. Top industries with the highest number of job postings
plt.figure(figsize=(10, 8))
top_industries = df['industry'].value_counts().head(10)
top_industries.plot(kind='barh').invert_yaxis()
plt.title('Top 10 Industries with the Highest Number of Job Postings')
plt.xlabel('Number of Job Postings')
plt.ylabel('Industry')
plt.show()

"""# What is the relationship between employment type and fraudulent job postings?"""

# 4. Relationship between employment type and fraudulent job postings
employment_type_fraud = df.groupby(['employment_type', 'fraudulent']).size().unstack()
employment_type_fraud.plot(kind='pie', subplots=True, figsize=(16, 8), autopct='%1.1f%%')
plt.title('Employment Type and Fraudulent Job Postings')
plt.ylabel('')
plt.show()

"""# How do the requirements of job postings (presence or absence of specified requirements) relate to their authenticity?"""

# 5. Histogram of the number of requirements in job postings
# For simplicity, let's consider 'requirements' text length as a proxy for the number of requirements
df['requirements_length'] = df['requirements'].str.len()
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='requirements_length', hue='fraudulent', element='step', bins=30)
plt.title('Job Posting Requirements Length and Authenticity')
plt.xlabel('Length of Requirements')
plt.ylabel('Count')
plt.show()

"""#  Data Preparation"""

df = df[['description', 'fraudulent']].dropna()
df['description'] = df['description'].astype(str)

# Check the prepared data
df.head()

"""#  Text Preprocessing

#### Perform tokenization, removal of stop words, and other text normalization steps.
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Tokenization and cleaning
def preprocess_text(text):
    # Convert to lowercase, tokenize, and remove stopwords
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return " ".join(filtered_tokens)

df['cleaned_description'] = df['description'].apply(preprocess_text)

nltk.download('stopwords')

df.head()

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Generate a word cloud for all job postings
all_text = ' '.join(df['cleaned_description'])
wordcloud_all = WordCloud(width=800, height=400).generate(all_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_all, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud for All Job Postings")
plt.show()

# Generate a word cloud for fraudulent job postings
fraud_text = ' '.join(df[df['fraudulent'] == 1]['cleaned_description'])
wordcloud_fraud = WordCloud(width=800, height=400).generate(fraud_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_fraud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud for Fraudulent Job Postings")
plt.show()

# Generate a word cloud for non-fraudulent job postings
non_fraud_text = ' '.join(df[df['fraudulent'] == 0]['cleaned_description'])
wordcloud_non_fraud = WordCloud(width=800, height=400).generate(non_fraud_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_non_fraud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud for Non-Fraudulent Job Postings")
plt.show()

from nltk import bigrams, trigrams
from collections import Counter

# Function to plot n-grams
def plot_ngrams(text_series, n=2):
    # Flatten list of words in the series and compute n-grams
    tokens = [token for row in text_series for token in row.split()]
    if n == 2:
        n_grams = bigrams(tokens)
    elif n == 3:
        n_grams = trigrams(tokens)
    else:
        raise ValueError("n needs to be 2 or 3 for bigrams or trigrams")

    # Count and display most common n-grams
    n_gram_freq = Counter(n_grams)
    most_common_ngrams = n_gram_freq.most_common(10)
    n_grams_df = pd.DataFrame(most_common_ngrams, columns=['n-gram', 'count'])

    # Plotting
    n_grams_df.plot(kind='bar', x='n-gram', y='count', title=f"Top 10 {n}-grams")
    plt.show()

# Plot bigrams
plot_ngrams(df['cleaned_description'], n=2)

# Plot trigrams
plot_ngrams(df['cleaned_description'], n=3)

"""# Term Frequency Distribution"""

from nltk import FreqDist

# Compute the frequency distribution of words
all_words = [word for text in df['cleaned_description'] for word in text.split()]
freq_dist = FreqDist(all_words)

# Plot the most common words
freq_dist.plot(30, title='Term Frequency Distribution')

"""#  Heatmap of Top Words in Fraudulent vs Non-Fraudulent Jobs"""

import numpy as np
import seaborn as sns

# Compute the top words in each category
top_n = 20
fraud_words = FreqDist([word for text in df[df['fraudulent'] == 1]['cleaned_description'] for word in text.split()])
non_fraud_words = FreqDist([word for text in df[df['fraudulent'] == 0]['cleaned_description'] for word in text.split()])

top_fraud = {word for word, count in fraud_words.most_common(top_n)}
top_non_fraud = {word for word, count in non_fraud_words.most_common(top_n)}

# Create a DataFrame with the frequency of top words in each category
top_words = top_fraud.union(top_non_fraud)
word_freq = pd.DataFrame({
    word: [fraud_words[word], non_fraud_words[word]] for word in top_words
}, index=['Fraudulent', 'Non-Fraudulent']).T

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(word_freq, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Heatmap of Top Words in Fraudulent vs Non-Fraudulent Jobs")
plt.show()

"""# Model Training"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Preparing the data
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['cleaned_description'])
y = df['fraudulent']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Decision Tree model
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

"""# Classification Report"""

from sklearn.metrics import classification_report

# Predicting on the test set
y_pred = decision_tree.predict(X_test)

# Printing the classification report
print(classification_report(y_test, y_pred))

"""#  Confusion Matrix"""

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Generating the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

from sklearn import tree

# Plotting the decision tree
plt.figure(figsize=(20,10))
tree.plot_tree(decision_tree, filled=True, feature_names=vectorizer.get_feature_names_out())
plt.title('Decision Tree Visualization')
plt.show()

"""#  ROC and AUC Plot"""

from sklearn.metrics import roc_curve, auc

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test, decision_tree.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

"""# Tuned Decision Tree"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

# Setting up the parameter grid for tuning
param_grid = {
    'max_depth': [10, 20,],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'criterion': ['gini', 'entropy']
}

# Initializing the decision tree classifier
dtree = DecisionTreeClassifier()

# Setting up the grid search
grid_search = GridSearchCV(estimator=dtree, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Training the model
grid_search.fit(X_train, y_train)

# Best parameters
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

# Best model
tuned_decision_tree = grid_search.best_estimator_

from sklearn.metrics import classification_report

# Predicting on the test set with the tuned model
y_pred_tuned = tuned_decision_tree.predict(X_test)

# Printing the classification report
print(classification_report(y_test, y_pred_tuned))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Generating the confusion matrix for the tuned model
conf_matrix_tuned = confusion_matrix(y_test, y_pred_tuned)

# Plotting the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_tuned, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Tuned Model')
plt.show()

from sklearn import tree

# Plotting the decision tree for the tuned model
plt.figure(figsize=(20,10))
tree.plot_tree(tuned_decision_tree, filled=True, feature_names=vectorizer.get_feature_names_out())
plt.title('Decision Tree Visualization for Tuned Model')
plt.show()

from sklearn.metrics import roc_curve, auc

# Compute ROC curve and ROC area for the tuned model
fpr_tuned, tpr_tuned, _ = roc_curve(y_test, tuned_decision_tree.predict_proba(X_test)[:, 1])
roc_auc_tuned = auc(fpr_tuned, tpr_tuned)

# Plotting the ROC curve for the tuned model
plt.figure()
plt.plot(fpr_tuned, tpr_tuned, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_tuned:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) for Tuned Model')
plt.legend(loc="lower right")
plt.show()

