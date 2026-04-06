import nltk
import string
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# ==============================
# 📥 Download NLTK Data
# ==============================
nltk.download('punkt')
nltk.download('stopwords')


# ==============================
# 🧹 Text Preprocessing Function
# ==============================
def preprocess(text):
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization
    words = word_tokenize(text)


    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    
    return " ".join(words)


# ==============================
# 📊 Dataset (Sample)
# ==============================
data = {
    "text": [
        "Win money now",
        "Hello how are you",
        "Free prize available",
        "Let's meet tomorrow",
        "Congratulations! You won a lottery",
        "Call me when you are free",
        "Limited offer! Buy now",
        "Are you coming today"
    ],
    "label": [
        "Spam", "Ham", "Spam", "Ham",
        "Spam", "Ham", "Spam", "Ham"
    ]
}

df = pd.DataFrame(data)


# ==============================
# 🧹 Apply Preprocessing
# ==============================
df['clean_text'] = df['text'].apply(preprocess)


# ==============================
# 🔢 Feature Extraction (TF-IDF)
# ==============================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])

y = df['label']


# ==============================
# 🔀 Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==============================
# 🤖 Model Training
# ==============================
model = MultinomialNB()
model.fit(X_train, y_train)


# ==============================
# 📊 Model Evaluation
# ==============================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n✅ Model Accuracy:", accuracy)


# ==============================
# 🔮 Prediction Function
# ==============================

def predict_spam(text):
    text = preprocess(text)
    vector = vectorizer.transform([text])
    result = model.predict(vector)[0]
    return result

# ==============================
# 🧪 Test Predictions
# ==============================
print("\n🔍 Testing Model:\n")

test_sentences = [
    "You won a free ticket",
    "Let's go for lunch",
    "Claim your prize now",
    "How are you doing today"
]

for sentence in test_sentences:
    print(f"Text:{sentence}")
    print("Prediction:", predict_spam(sentence))
    print("-" * 40)


# ==============================
# 🧑‍💻 User Input
# ==============================
while True:
    user_input = input("\nEnter a message (or type 'exit'): ")
    
    if user_input.lower() == 'exit':
        print("👋 Exiting...")
        break
    
    prediction = predict_spam(user_input)
    print("📌 Result:",prediction)