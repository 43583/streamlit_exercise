from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns


df = pd.read_csv("sentiment.csv")
df.dropna(inplace=True)

X = df['selected_text']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC())
])

text_clf.fit(X_train, y_train)
predictions = text_clf.predict(X_test)

cm = confusion_matrix(y_test, predictions)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', ax=ax, xticklabels=text_clf.classes_, yticklabels=text_clf.classes_)

st.title("Sentiment analyser")

form = st.form(key='sentiment-form')
user_input = form.text_area('Enter your text')
submit = form.form_submit_button('Submit')
df = classification_report(y_test, predictions, output_dict=True)

if submit:
    result = text_clf.predict([user_input])[0]
    sentimet = f'Sentiment is {result}.'

    if result == 'positive':
        st.badge(sentimet, color='green')
    elif result == 'negative':
        st.badge(sentimet, color='red')
    else:
        st.badge(sentimet, color='grey')

    st.dataframe(pd.DataFrame(df).transpose())
    st.pyplot(fig)
    