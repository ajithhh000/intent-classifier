import os, joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

X = [
    # Greeting (15)
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "hi there", "hello team", "hey folks", "greetings", "howdy", "welcome",
    "hey support", "hello agent", "hi customer service",
    
    # Question (15)
    "how to reset password", "what is my balance", "where is my order", 
    "how do I login", "payment methods", "delivery time", "refund policy",
    "cancel order", "track package", "account settings", "billing history",
    "change address", "subscription details", "loyalty points", "store hours",
    
    # Complaint (10)
    "cancel subscription", "broken product", "late delivery", "wrong item",
    "poor service", "not working", "refund now", "charge error", "delete account","bad quality",
    
    # Praise (10)
    "great service", "excellent support", "love product", "fast delivery",
    "amazing quality", "highly recommend", "best ever", "thank you team",
    "perfect order", "outstanding service"
]
y = ["greeting"]*15 + ["question"]*15 + ["complaint"]*10 + ["praise"]*10


pipeline=Pipeline([("vect",CountVectorizer()),("clf",MultinomialNB())])
pipeline.fit(X,y)

os.makedirs("model/artifacts", exist_ok=True)
joblib.dump(pipeline, "model/artifacts/intent_model.pkl")
print("trained")
