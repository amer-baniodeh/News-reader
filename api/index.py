import os
import joblib
import feedparser
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load PKL models from root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "svm_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

class TextPayload(BaseModel):
    text: str

class FeedPayload(BaseModel):
    url: str

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/predict")
def predict_category(payload: TextPayload):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    transformed_text = vectorizer.transform([payload.text])
    prediction = model.predict(transformed_text)[0]
    
    return {"text": payload.text, "prediction": str(prediction)}

@app.post("/api/parse-feed")
def parse_feed(payload: FeedPayload):
    feed = feedparser.parse(payload.url)
    if not feed.entries:
        raise HTTPException(status_code=400, detail="Could not fetch or parse RSS feed")
    
    results = []
    for entry in feed.entries[:10]: # Process top 10 articles
        title = entry.get('title', '')
        summary = entry.get('summary', '')
        text_to_classify = f"{title} {summary}"
        
        transformed_text = vectorizer.transform([text_to_classify])
        pred = model.predict(transformed_text)[0]
        
        results.append({
            "title": title,
            "link": entry.get('link', '#'),
            "prediction": str(pred)
        })
        
    return {"articles": results}