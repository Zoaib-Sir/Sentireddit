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
- Preprocess text by removing links, markdown, emojis, and filtering out non-English content
- Processes up to 500 comments at once for large-scale analysis
- Sentiment analysis powered by RoBERTa model fine-tuned on Twitter data
- Emotion classification using a DistilRoBERTa-based emotion detection model
- Dynamic pie chart for Negative, Neutral, and Positive sentiment distribution
- Bar chart showing relative proportions of emotions such as Anger, Fear, Joy, Sadness, Surprise, and more
- Responsive frontend built with HTML, CSS, and vanilla JavaScript

## MACHINE LEARNING DETAILS

- Sentiment analysis is performed using a Transformers pipeline with the `cardiffnlp/twitter-roberta-base-sentiment` model.
- The model returns discrete sentiment labels (LABEL_0, LABEL_1, LABEL_2), which are aggregated to compute sentiment distribution.
- Emotion classification is performed using the `j-hartmann/emotion-english-distilroberta-base` model via Transformers pipeline.
- Comments are processed in bulk by passing the entire cleaned list to the respective Transformer pipelines.
- Preprocessing removes noise (links, markdown), demojizes emojis, and filters non-English text using `regex`, `emoji`, and `langdetect`.
- Chart generation uses Matplotlib to produce Base64-encoded PNGs, which are embedded directly in the frontend's JSON response.

## ARCHITECTURE OVERVIEW

1. **Flask Application (`app.py`)**
   - Serves the HTML frontend and handles POST requests with the Reddit URL
   - Uses `PRAW` to fetch up to 500 top-level Reddit comments
   - Applies preprocessing: cleans markdown, removes links/emojis, filters out non-English text
   - Loads sentiment and emotion Transformer pipelines (cached at startup for performance)
   - Runs both models on the cleaned comments and computes label distributions
   - Generates sentiment and emotion charts as Base64-encoded PNGs and returns them in a JSON response

2. **Machine Learning Integration**
   - Sentiment analysis uses the `cardiffnlp/twitter-roberta-base-sentiment` model
   - Emotion detection uses the `j-hartmann/emotion-english-distilroberta-base` model
   - Both models are run directly within the main Flask app without microservice separation

3. **Frontend Static Files**
   - `index.html` provides the UI with a search box and chart display containers
   - `style.css` styles the layout, charts, error messages, and loading spinner
   - JavaScript logic in `index.html` handles user interaction, sends POST requests, and dynamically updates the charts

## SETUP AND INSTALLATION

1. **Clone the repository**
    ```bash
    git clone https://github.com/Zoaib-Sir/Sentireddit.git
    cd Sentireddit
    ```

2. **Create a virtual environment and install dependencies**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3. **Configure environment variables**

    Create a `.env` file in the root folder with the following content:
    ```env
    REDDIT_CLIENT_ID=your_reddit_client_id
    REDDIT_CLIENT_SECRET=your_reddit_client_secret
    REDDIT_USER_AGENT=your_user_agent
    FLASK_DEBUG=True
    ```

4. **Run the application**
    ```bash
    python app.py
    ```

5. **Access the application**

    Open your browser and go to:
    ```
    http://localhost:5000
    ```

    Enter any valid Reddit post URL to analyze its comments.


## FUTURE IMPROVEMENTS

- Add authentication and rate limiting to prevent abuse
- Support multiple languages by integrating multilingual sentiment/emotion models
- Improve error handling and display more detailed feedback to users
- Deploy the application with autoscaling support using platforms like Render or Railway
- Convert to a microservice-based architecture for better modularity and scalability
- Enhance the frontend with modern frameworks (like React or Vue) and add interactive visualizations (e.g., Plotly, Chart.js)
- Include additional analytics like toxicity detection or topic modeling
- 
