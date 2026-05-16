import pandas as pd
import nltk
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# load dataset
df = pd.read_csv("Task_2/tickets.csv")

# remove missing values
df = df.dropna(subset=['Ticket Description', 'Ticket Type', 'Ticket Priority'])

# text cleaning
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# apply cleaning
df['clean_text'] = df['Ticket Description'].apply(clean_text)

# convert text to numerical features
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = vectorizer.fit_transform(df['clean_text'])

# target variables
y_category = df['Ticket Type']
y_priority = df['Ticket Priority']

# split data
X_train, X_test, y_train_cat, y_test_cat = train_test_split(
    X, y_category, test_size=0.2, random_state=42
)

_, _, y_train_pri, y_test_pri = train_test_split(
    X, y_priority, test_size=0.2, random_state=42
)

# train models
model_cat = LogisticRegression(max_iter=200)
model_cat.fit(X_train, y_train_cat)

model_pri = LogisticRegression(max_iter=200)
model_pri.fit(X_train, y_train_pri)

# predictions
pred_cat = model_cat.predict(X_test)
pred_pri = model_pri.predict(X_test)

# results
print("Category Accuracy:", accuracy_score(y_test_cat, pred_cat))
print(classification_report(y_test_cat, pred_cat))

print("\nPriority Accuracy:", accuracy_score(y_test_pri, pred_pri))
print(classification_report(y_test_pri, pred_pri))

# sample test
sample = ["My internet is not working and I need urgent help"]

clean_sample = [clean_text(t) for t in sample]
vector_sample = vectorizer.transform(clean_sample)

print("\nSample Prediction:")
print("Category:", model_cat.predict(vector_sample)[0])
print("Priority:", model_pri.predict(vector_sample)[0])