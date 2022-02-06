from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
import re
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

## init Flask App
app = Flask(__name__)

# Load Pickle model
with open('model','rb') as f:
  model=pickle.load(f)

# We need to fit the TFIDF VEctorizer
news_dataset = pd.read_csv('train.csv')
news_dataset.shape
# print the first 5 rows of the dataframe
news_dataset.head()
# counting the number of missing values in the dataset
news_dataset.isnull().sum()

# replacing the null values with empty string
news_dataset = news_dataset.fillna('')

# merging the author name and news title
news_dataset['content'] = news_dataset['author']+' '+news_dataset['title']
# separating the data & label
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']

port_stem = PorterStemmer()

def stemming(content):      #Creating function named stemming
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)    #Removes numbers and other characters not included in the alphabet to be fed to the content
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming) #Processing for stemming and making passing them into the content

#separating the data and label
X = news_dataset['content'].values
Y = news_dataset['label'].values

# converting the textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)

model = LogisticRegression()

model.fit(X_train, Y_train)

def fake_news_detect(content):
  X_new = X_test[0]

  prediction = model.predict(X_new)
  return(prediction)

print(Y_test[0])

def fake_news(content):
  X_new = X_test[X]
  




# Defining the site route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
  if request.method == 'POST':
    message = request.form['message']
    pred = fake_news_detect(message)
    print(pred)
    return render_template('index.html', prediction=pred)
  else:
    return render_template('index.html', prediction="Something went wrong")
      



if __name__ == '__main__':
    app.run(debug=True)