from flask import Flask, render_template, request
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from openai import AzureOpenAI
import os
from dotenv import load_dotenv


load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-05-15",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")


captions = [
    "a red floral dress", "a sleeveless cotton top", "a casual striped t-shirt","a beautiful blazer",
    "a traditional kurta for women", "a fashionable blouse", "a white embroidered top"
]

@app.route("/", methods=["GET", "POST"])
def index():
    description = None
    filename = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            image = Image.open(filename).convert("RGB")

            
            image_inputs = clip_processor(images=image, return_tensors="pt")
            image_features = clip_model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            
            text_inputs = clip_processor(text=captions, return_tensors="pt", padding=True)
            text_features = clip_model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarity = torch.matmul(image_features, text_features.T)
            best_caption = captions[similarity.argmax()]

            
            prompt = f"This product is {best_caption}. It is"
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that describes fashion products."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.8,
                top_p=0.95
            )

            description = response.choices[0].message.content.strip()

    return render_template("index.html", description=description, filename=filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

