from flask import Flask, render_template, request, jsonify
import praw
import re
import emoji
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from langdetect import detect
from transformers import pipeline
import nltk
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Model cache setup
MODEL_CACHE = {
    'sentiment': None,
    'emotion': None
}

def load_models():
    """Load and cache models during app initialization"""
    if not MODEL_CACHE['sentiment']:
        MODEL_CACHE['sentiment'] = pipeline(
            "sentiment-analysis", 
            model="cardiffnlp/twitter-roberta-base-sentiment"
        )
    
    if not MODEL_CACHE['emotion']:
        MODEL_CACHE['emotion'] = pipeline(
            "text-classification", 
            model="j-hartmann/emotion-english-distilroberta-base"
        )
    print("✅ Models loaded and cached")

# Load models immediately when app starts
load_models()

# Initialize Reddit client
reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent=os.getenv('REDDIT_USER_AGENT')
)

def preprocess_comments(comments):
    cleaned = []
    for comment in comments:
        text = re.sub(r'\[.*?\]\(.*?\)|http\S+|\W', ' ', comment)
        text = emoji.demojize(text).replace(':', ' ')
        try:
            if detect(text) != 'en':
                continue
        except:
            continue
        cleaned.append(text.strip().lower())
    return cleaned

def generate_charts(sentiment_data, emotion_data):
    # Sentiment Pie Chart
    plt.figure(figsize=(6, 6))
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    sentiment_values = [sentiment_data.get('LABEL_0', 0), 
                       sentiment_data.get('LABEL_1', 0), 
                       sentiment_data.get('LABEL_2', 0)]
    plt.pie(sentiment_values, labels=sentiment_labels, autopct='%1.1f%%',
            colors=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    plt.title('Sentiment Distribution')
    img1 = BytesIO()
    plt.savefig(img1, format='png')
    plt.close()
    img1.seek(0)
    sentiment_url = base64.b64encode(img1.getvalue()).decode('utf-8')

    # Emotion Bar Chart
    plt.figure(figsize=(8, 4))
    emotions = sorted(emotion_data.items(), key=lambda x: x[1], reverse=True)
    plt.bar([e[0].title() for e in emotions], [e[1] for e in emotions], color='#6c5ce7')
    plt.title('Emotional Tone Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    img2 = BytesIO()
    plt.savefig(img2, format='png')
    plt.close()
    img2.seek(0)
    emotion_url = base64.b64encode(img2.getvalue()).decode('utf-8')

    return sentiment_url, emotion_url

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        post_url = request.form.get('url', '').strip()
        
        # Validate URL format
        if not post_url.startswith(('https://www.reddit.com/', 'https://reddit.com/')):
            return jsonify({'error': 'Invalid Reddit URL format. Example: https://www.reddit.com/r/subreddit/comments/...'})
            
        try:
            submission = reddit.submission(url=post_url)
            submission.id  # Validate post exists
        except Exception as e:
            return jsonify({'error': f'Reddit post not found: {str(e)}'})

        try:
            submission.comments.replace_more(limit=0)
            comments = [c.body for c in submission.comments.list()[:500]]
            
            if not comments:
                return jsonify({'error': 'No comments found in this post'})
            
            cleaned_comments = preprocess_comments(comments)
            
            # Sentiment analysis
            sentiment_results = MODEL_CACHE['sentiment'](cleaned_comments)
            sentiments = [res['label'] for res in sentiment_results]
            sentiment_dist = pd.Series(sentiments).value_counts(normalize=True).to_dict()
            
            # Emotion analysis
            emotion_results = MODEL_CACHE['emotion'](cleaned_comments)
            emotions = [res['label'] for res in emotion_results]
            emotion_dist = pd.Series(emotions).value_counts(normalize=True).to_dict()
            
            # Generate charts
            sentiment_img, emotion_img = generate_charts(sentiment_dist, emotion_dist)
            
            return jsonify({
                'sentiment': sentiment_dist,
                'emotion': emotion_dist,
                'sentiment_chart': sentiment_img,
                'emotion_chart': emotion_img
            })
            
        except Exception as e:
            return jsonify({'error': f'Analysis failed: {str(e)}'})
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=os.getenv('FLASK_DEBUG', False))