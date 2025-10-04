from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import InferenceClient

basedir = os.path.abspath(os.path.dirname(__file__))
public_folder = os.path.join(basedir, 'public')

app = Flask(__name__, 
            template_folder=public_folder, 
            static_folder=public_folder) 

CORS(app) 

@app.route("/")
def serve_frontend():
    return render_template("index.html")

@app.route('/<path:filename>')
def serve_static(filename):
    # This route will serve any file requested from the public folder.
    # For example, if you request /css/style.css, it will look in public/css/style.css
    return send_from_directory(public_folder, filename)

client = InferenceClient(
    provider="featherless-ai",
    api_key=os.environ["HF_API_KEY"],
)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        os.makedirs('./uploads', exist_ok=True)
        
        model_path = "eye_disease_model.pth"
        classes_path = "classes.json"
        
        if not os.path.exists(model_path) or not os.path.exists(classes_path):
            return jsonify({'error': 'Required model files are missing'}), 500

        query = request.form.get('query')

        if not query:
            return jsonify({'error': 'No query provided'}), 400
            
        pdf = request.files.get('pdf')
        xray = request.files.get('xray')
        eye_image = request.files.get('eyeImage')

        if pdf and pdf != 'no_image_data':
            pdf.save(f"./uploads/{pdf.filename}")
            
        if xray and xray != 'no_image_data':
            xray.save(f"./uploads/{xray.filename}")
            
        eye_disease_prediction = 'N/A'
        if eye_image and eye_image != 'no_image_data':
            try:
                upload_path = f"./uploads/{eye_image.filename}"
                eye_image.save(upload_path)
                if not os.path.exists(upload_path):
                    raise FileNotFoundError(f"Failed to save image: {upload_path}")
            except Exception as e:
                return jsonify({'error': f'Error processing eye image: {str(e)}'}), 500

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            with open("classes.json", "r") as f:
                classes = json.load(f)
                
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, len(classes))

            model.load_state_dict(torch.load("eye_disease_model.pth", map_location=device))
            model = model.to(device)
            model.eval()

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

            def predict_image(img_path):
                image = Image.open(img_path).convert("RGB")
                image = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(image)
                    _, predicted = torch.max(output, 1)
                return classes[predicted.item()]

            eye_img = f"./uploads/{eye_image.filename}"            
            eye_disease_prediction = predict_image(eye_img)
            print("Prediction for", eye_img, ":", eye_disease_prediction)

        structured_prompt = f"""
eye_disease: {eye_disease_prediction} 
Based on this query: {query}
Provide medical advice in this exact format:

### Key Suggestions for Self-Care
- Stay hydrated with warm fluids
- Use honey for cough (adults and children >1 year)
 
### Lifestyle Modifications
- Use a humidifier in bedroom
- Avoid irritants and allergens
 
### When to Seek Medical Attention
- If symptoms worsen or persist >7 days
- If experiencing severe symptoms
 
### Warnings & Precautions
- Note about OTC medication safety
- When to consult healthcare provider

Keep responses evidence-based and practical.
Use proper Markdown line breaks between sections.
"""

        completion = client.chat.completions.create(
            model="Intelligent-Internet/II-Medical-8B-1706",
            messages=[{"role": "user", "content": structured_prompt}],
        )

        response_text = completion.choices[0].message.content
        cleaned_response = '\n'.join(
            line for line in response_text.splitlines()
            if not line.strip().startswith('<') and not line.strip().endswith('>')
        )

        response = {
            'status': 'success',
            'response': cleaned_response
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
