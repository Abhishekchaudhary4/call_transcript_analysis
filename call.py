import streamlit as st
import tempfile
import os
import google.generativeai as genai
from textblob import TextBlob
from dotenv import load_dotenv

load_dotenv()

# Configure Google API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def analyze_sentiment(text):
    """Analyze the sentiment of the given text."""
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return "Positive", sentiment
    elif sentiment < 0:
        return "Negative", sentiment
    else:
        return "Neutral", sentiment

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary file and return the path."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error handling uploaded file: {e}")
        return None

# Streamlit app interface
st.title('Call Sentiment Analysis App')

with st.expander("About this app"):
    st.write("""
        This app analyzes the sentiment of call history text documents. 
        Upload a text file containing call chat history to get sentiment analysis.
    """)

text_file = st.file_uploader("Upload Call History Text File", type=['txt'])
if text_file is not None:
    text_path = save_uploaded_file(text_file)
    
    if st.button('Analyze Sentiment'):
        with st.spinner('Analyzing...'):
            with open(text_path, 'r') as file:
                content = file.read()
            
            sentiment, score = analyze_sentiment(content)
            
            st.subheader("Sentiment Analysis Results")
            st.write(f"Overall Sentiment: {sentiment}")
            st.write(f"Sentiment Score: {score:.2f}")
            
            # Visualize sentiment score
            st.progress((score + 1) / 2)  # Normalize score to 0-1 range
            
            # Display word cloud (optional)
            try:
                from wordcloud import WordCloud
                import matplotlib.pyplot as plt
                
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(content)
                
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            except ImportError:
                st.write("Install wordcloud package to see word cloud visualization.")

# Clean up temporary file
if 'text_path' in locals():
    os.unlink(text_path)