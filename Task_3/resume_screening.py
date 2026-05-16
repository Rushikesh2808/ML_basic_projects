import pandas as pd
import string
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("resumes.csv")

# Job Description
job_description = """
Python SQL Machine Learning Data Analysis Power BI
"""

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

# Clean resumes
df['clean_resume'] = df['resume'].apply(clean_text)

# Clean JD
clean_jd = clean_text(job_description)

# Combine all text
documents = [clean_jd] + df['clean_resume'].tolist()

# Convert text to vectors
vectorizer = CountVectorizer()
matrix = vectorizer.fit_transform(documents)

# Similarity scores
similarity = cosine_similarity(matrix[0:1], matrix[1:])

# Add scores
df['score'] = similarity.flatten() * 100

# Skill extraction
required_skills = set(clean_jd.split())

missing_skills_list = []

for resume in df['clean_resume']:
    resume_skills = set(resume.split())
    missing = required_skills - resume_skills
    missing_skills_list.append(", ".join(missing))

df['missing_skills'] = missing_skills_list

# Rank candidates
ranked = df.sort_values(by='score', ascending=False)

# Display results
print("\nCandidate Ranking:\n")
print(ranked[['name', 'score', 'missing_skills']])

# Best candidate
best = ranked.iloc[0]

print("\nBest Candidate:")
print(best['name'])

print("\nMatch Score:")
print(round(best['score'], 2), "%")

print("\nMissing Skills:")
print(best['missing_skills'])