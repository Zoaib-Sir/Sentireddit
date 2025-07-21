# SENTIREDDIT

Sentireddit is a web application that performs Sentiment and Emotion Analysis on Reddit comments. By entering a Reddit post URL, users can view graphical distributions of sentiment polarity and emotional tone expressed by the community. The project uses two separate Flask microservices to run Machine Learning pipelines for sentiment detection and emotion classification.

## SCREENSHOTS
![Screenshot 2025-05-03 210504](https://github.com/user-attachments/assets/42bdc1a9-bfe1-4d4d-addd-00469219a931)
![Screenshot 2025-05-03 210519](https://github.com/user-attachments/assets/b8febfb0-ff2b-4a01-9c73-c82706bdba30)
![Screenshot 2025-05-01 181837](https://github.com/user-attachments/assets/cebc8bc8-be30-4896-b4c4-d20f96a44e4f)

## Deployment

Access the live demo here: [https://sentireddit.onrender.com](https://sentireddit.onrender.com)

## FEATURES

- Extract comments from any public Reddit post using PRAW
- Preprocess text by removing links, Markdown, emojis, and filtering for English
- Batch requests for efficient analysis of large numbers of comments
- Sentiment Analysis service powered by DistilBERT fine-tuned on SST-2 dataset
- Emotion Classification service using a DistilBERT-based emotion detection model
- Dynamic pie chart for Negative, Neutral, and Positive sentiment breakdown
- Bar chart showing relative proportions of emotions such as Anger, Fear, Joy, Sadness, and Surprise
- Responsive frontend with HTML, CSS, and vanilla JavaScript

## MACHINE LEARNING DETAILS

- Sentiment API uses a Transformers pipeline for text classification with the DistilBERT-base-uncased-SST-2 model.
- Raw probabilities are converted into three classes (Negative, Neutral, Positive) using threshold logic.
- Emotion API uses a Transformers pipeline with the `bhadresh-savani/distilbert-base-uncased-emotion` model.
- Both services handle batch requests of up to 50 texts per call for efficiency.
- Preprocessing removes noise (links, Markdown), demojizes emojis, and filters non-English text using regex, `emoji`, and `langdetect`.
- Chart generation uses Matplotlib to save figures as Base64-encoded PNGs, which are embedded directly in the frontend JSON response.

## ARCHITECTURE OVERVIEW

1. **Flask Frontend (app.py)**
   - Renders the template and handles POST requests with the Reddit URL
   - Uses PRAW to fetch up to 500 comments
   - Applies preprocessing, filtering, and language detection
   - Sends batches of comments to the two ML microservices
   - Receives label distributions and returns Base64-encoded chart images in JSON

2. **Sentiment Microservice**
   - Flask API wrapping a Transformers pipeline with DistilBERT-base-uncased fine-tuned on SST-2
   - Maps raw scores into Negative, Neutral, or Positive categories

3. **Emotion Microservice**
   - Flask API using the `bhadresh-savani/distilbert-base-uncased-emotion` model
   - Produces probability scores for emotional classes and returns the top label per text

4. **Frontend Static Files**
   - `index.html` and `style.css` for layout and design
   - Vanilla JavaScript to POST the Reddit URL and update chart images dynamically

## SETUP AND INSTALLATION

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/sentireddit.git
    cd sentireddit
    ```

2. Create a virtual environment and install dependencies:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3. Configure environment variables:
    - Create a `.env` file in the root folder with your Reddit credentials (`REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`)
    - Optionally set `SENTIMENT_API_URL` and `EMOTION_API_URL` if services are deployed remotely

4. Run the microservices and frontend:
    - **Sentiment API**: `flask run` in the sentiment service folder (or set `FLASK_APP=sentiment_api.py`)
    - **Emotion API**: `flask run` in the emotion service folder
    - **Frontend**: `python app.py` in the main project folder

5. Open your browser at `http://localhost:5000` and enter a Reddit post URL.

## FUTURE IMPROVEMENTS

- Add authentication and rate limiting for API endpoints
- Support multiple languages with multilingual models
- Implement real-time streaming updates to build charts live
- Deploy microservices to cloud platforms with autoscaling
- Enhance the frontend with modern frameworks and interactive visualizations
