from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import clip
import clip  # Make sure this is installed correctly
import torch
from PIL import Image
import pymongo
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# MongoDB setup
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["dating_profile_db"]
profiles = db["profiles"]

# Load CLIP model for image analysis
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load Hugging Face model for text generation
generator = pipeline("text-generation", model="EleutherAI/gpt-j-6B")

@app.route("/analyze", methods=["POST"])
def analyze_profile():
    data = request.json

    name = data.get("name", "User")
    goal = data.get("goal", "Dating")
    region = data.get("region", "Global")
    interests = data.get("interests", [])
    personality = data.get("personality", "Outgoing")
    uploaded_images = data.get("images", [])

    # Generate a dating bio using GPT-J
    bio_prompt = f"Write a fun dating profile bio for {name}, who is looking for {goal} in {region}. They enjoy {', '.join(interests)} and have a {personality} personality."
    bio = generate_text(bio_prompt)

    # Analyze profile images
    image_feedback = analyze_images(uploaded_images)

    # Save to MongoDB
    profile_data = {
        "name": name,
        "goal": goal,
        "region": region,
        "interests": interests,
        "personality": personality,
        "bio": bio,
        "image_feedback": image_feedback
    }
    profiles.insert_one(profile_data)

    return jsonify({
        "bio": bio,
        "image_feedback": image_feedback,
        "message": "Profile analyzed successfully!"
    })

def generate_text(prompt):
    """Generate text using Hugging Face GPT-J"""
    result = generator(prompt, max_length=100, num_return_sequences=1)
    return result[0]["generated_text"]

def analyze_images(image_paths):
    """Analyze images using CLIP"""
    results = []
    for img_path in image_paths:
        try:
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                _ = model.encode_image(image)  # Extract features

            feedback = "Great image choice!" if "smiling" in img_path else "Try using a clearer photo."
            results.append({"image": img_path, "feedback": feedback})

        except Exception as e:
            results.append({"image": img_path, "feedback": f"Error processing image: {str(e)}"})

    return results

if __name__ == "__main__":
    app.run(debug=True)