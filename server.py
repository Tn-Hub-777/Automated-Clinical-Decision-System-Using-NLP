from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import send_from_directory
import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import InferenceClient

@app.route("/")
def serve_frontend():
    return send_from_directory("frontend/build", "index.html")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the Hugging Face client
client = InferenceClient(
    provider="featherless-ai",
    api_key=os.environ["HF_API_KEY"],
)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs('./uploads', exist_ok=True)
        
        # Ensure model files exist before processing
        model_path = "eye_disease_model.pth"
        classes_path = "classes.json"
        
        if not os.path.exists(model_path) or not os.path.exists(classes_path):
            return jsonify({'error': 'Required model files are missing'}), 500

        query = request.form.get('query')

        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Get files or default value
        pdf = request.files.get('pdf')
        xray = request.files.get('xray')
        eye_image = request.files.get('eyeImage')

        # Process files if they exist
        if pdf and pdf != 'no_image_data':
            # Process PDF file
            pdf.save(f"./uploads/{pdf.filename}")
            
        if xray and xray != 'no_image_data':
            # Process X-ray image
            xray.save(f"./uploads/{xray.filename}")
            
        if eye_image and eye_image != 'no_image_data':
            # Process eye image
            try:
                upload_path = f"./uploads/{eye_image.filename}"
                eye_image.save(upload_path)
                if not os.path.exists(upload_path):
                    raise FileNotFoundError(f"Failed to save image: {upload_path}")
            except Exception as e:
                return jsonify({'error': f'Error processing eye image: {str(e)}'}), 500

            # Set device (GPU if available)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load classes
            with open("classes.json", "r") as f:
                classes = json.load(f)
            
            # Rebuild the model architecture (ResNet18 with custom final layer)
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, len(classes))

            # Load the saved weights
            model.load_state_dict(torch.load("eye_disease_model.pth", map_location=device))
            model = model.to(device)
            model.eval()

            #print("✅ Model and classes loaded successfully!")
            #print("Classes:", classes)

            # Define the image transform (same as in your notebook)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

            # Prediction function
            def predict_image(img_path):
                # Load and preprocess the image
                image = Image.open(img_path).convert("RGB")
                image = transform(image).unsqueeze(0).to(device)

                # Run inference            
                with torch.no_grad():
                    output = model(image)
                    _, predicted = torch.max(output, 1)

                # Return the predicted class
                return classes[predicted.item()]

            # Predict on a specific eye image
            eye_img = f"./uploads/{eye_image.filename}" 
            eye_disease_prediction = predict_image(eye_img)
            print("Prediction for", eye_img, ":", eye_disease_prediction)

        structured_prompt = f"""
        eye_disease: {eye_disease_prediction if eye_image and eye_image != 'no_image_data' else 'N/A'} \n
        
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
        # Preserve line breaks in markdown
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
