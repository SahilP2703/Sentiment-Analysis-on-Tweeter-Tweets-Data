import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Load the pre-trained model and the vectorizer from the pickle files
with open('tweet_sentiment_analysis.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('count_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define a function to predict sentiment
def predict_sentiment(text):
    # Transform the input text using the loaded vectorizer
    transformed_text = vectorizer.transform([text])
    # Predict the sentiment using the loaded model
    prediction = model.predict(transformed_text.reshape(1, -1))[0]
    # Return 'Positive' for 1 and 'Negative' for 0

    sentiment_mapping = {
    'Positive': 'Positive sentiment',
    'Neutral': 'Neutral sentiment',
    'Negative': 'Negative sentiment'
    }

    predicted_sentiment_label = sentiment_mapping[prediction]

    return predicted_sentiment_label

# Streamlit app
st.title("Tweet Sentiment Analyzer")
st.write("Enter a tweet to analyze its sentiment.")

# Input text from user
user_input = st.text_area("Tweet Text")

if st.button("Analyze Sentiment"):
    if user_input:
        # Predict sentiment
        sentiment = predict_sentiment(user_input)
        # Display the result
        st.write(f"Sentiment: {sentiment}")

        #happ = "images\happiness.png"
        if sentiment == 'Positive sentiment':
            st.image('images\happiness.png', width=50)
        elif sentiment == 'Negative sentiment':
            st.image('images\sad.png', width=50)
        elif sentiment == 'Neutral sentiment':
            st.image('images\neutral.png', width=50)
    else:
        st.write("Please enter some text to analyze.")

# Run the app
if __name__ == '__main__':
    st.set_option('deprecation.showfileUploaderEncoding', False)
