import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
import re

# Step 2: Load and preprocess the data
data = pd.read_csv('labeled_data.csv')  # Replace 'labeled_data.csv' with your dataset file


special_character_remover = re.compile('[/(){}\[\]\|@,;]')
extra_symbol_remover = re.compile('[^0-9a-z ]')
STOPWORDS = set(stopwords.words('english'))

## cleaning data 

def clean_text(text):
    text = text.lower()
    text = special_character_remover.sub(' ', text)
    text = extra_symbol_remover.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text
data['name'] = data['name'].apply(clean_text)

data['brand'] = data['brand'].str.lower() 
data['category'] = data['category'].str.lower() 

## setting up classification trainer 
X = data['name']+" "+data['brand']
y = data['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #train on 6 , test on 3


lr = Pipeline([ 
                ('tfidf', TfidfVectorizer()),
                ('clf', LogisticRegression()),
            ])

lr.fit(X_train, y_train)

## checking predicting results
y_pred1 = lr.predict(X_test)



## printing accuracy results
print(f"accuracy is: {accuracy_score(y_pred1, y_test)}")
print(f"classification accuracy is : {classification_report(y_test, y_pred1)}")

