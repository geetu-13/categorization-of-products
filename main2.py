import pandas as pd
from sklearn.metrics import accuracy_score


# Load the data from productSample.csv
df_product_sample = pd.read_csv("labeled_data.csv")

# Load the data from test.csv
df_test = pd.read_csv("test.csv")

X_train = df_product_sample['name'] + " " + df_product_sample['brand']
X_test = df_test['name'] + " " + df_test['brand']


from sklearn.preprocessing import LabelEncoder

# Assuming "category.csv" has a column: "category_name"
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(df_product_sample['category'])

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

classifier = OneVsRestClassifier(SVC(kernel='linear'))
classifier.fit(X_train_vec, y_train)


y_pred1 =  classifier.predict(X_train_vec)
print(f"accuracy is: {accuracy_score(y_pred1, y_train)}")


y_pred = classifier.predict(X_test_vec)

y_pred_category_names = label_encoder.inverse_transform(y_pred)

print(y_pred_category_names)
