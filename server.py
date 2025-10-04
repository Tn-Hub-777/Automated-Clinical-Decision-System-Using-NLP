from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import InferenceClient
import sys 

# Determine the absolute path of the directory containing this server.py file
# When running on Render, this will likely be /opt/render/project/src/
current_file_dir = os.path.abspath(os.path.dirname(__file__))

# Go up one level from 'src' to get to the project's root directory on Render
# This should be /opt/render/project/
project_root_on_render = os.path.dirname(current_file_dir)

# Define the path to your 'public' folder, relative to the project root on Render
# This should be /opt/render/project/public/
public_folder = os.path.join(project_root_on_render, 'public')

print(f"DEBUG: Current file directory (where server.py is): {current_file_dir}", file=sys.stderr)
print(f"DEBUG: Calculated project root on Render: {project_root_on_render}", file=sys.stderr)
print(f"DEBUG: Public folder path: {public_folder}", file=sys.stderr)
print(f"DEBUG: Does public folder exist: {os.path.exists(public_folder)}", file=sys.stderr)
print(f"DEBUG: Contents of public folder: {os.listdir(public_folder) if os.path.exists(public_folder) else 'Does not exist'}", file=sys.stderr)

# Initialize Flask app
# - template_folder: tells Flask where to find HTML templates (e.g., index.html)
# - static_folder: tells Flask where to find static assets (e.g., CSS, JS, images)
app = Flask(__name__, 
            template_folder=public_folder, 
            static_folder=public_folder) # Point static files to public_folder

CORS(app) 

# --- HUGGING FACE CLIENT INITIALIZATION ---
try:
    hf_api_key = os.environ.get("HF_API_KEY")
    if not hf_api_key:
        print("ERROR: HF_API_KEY environment variable is not set!", file=sys.stderr)
        sys.exit("HF_API_KEY environment variable is not set, cannot initialize Hugging Face client.")
    client = InferenceClient(
        provider="featherless-ai",
        api_key=hf_api_key, 
    )
    print("DEBUG: Hugging Face client initialized successfully.", file=sys.stderr)
except Exception as e:
    print(f"ERROR: Failed to initialize Hugging Face client: {e}", file=sys.stderr)
    sys.exit(f"Failed to initialize Hugging Face client: {e}") 

# --- MODEL AND CLASSES PATHS ---
# Assuming these are also in the actual project root on Render
model_path = os.path.join(project_root_on_render, "eye_disease_model.pth")
classes_path = os.path.join(project_root_on_render, "classes.json")

print(f"DEBUG: Model path: {model_path}", file=sys.stderr)
print(f"DEBUG: Classes path: {classes_path}", file=sys.stderr)
print(f"DEBUG: Model file exists: {os.path.exists(model_path)}", file=sys.stderr)
print(f"DEBUG: Classes file exists: {os.path.exists(classes_path)}", file=sys.stderr)

# --- ROUTES ---
@app.route("/")
def serve_frontend():
    print("DEBUG: Serving index.html request.", file=sys.stderr)
    try:
        return render_template("index.html")
    except Exception as e:
        print(f"ERROR: Failed to render index.html: {e}", file=sys.stderr)
        return f"Internal Server Error during template rendering: {e}", 500

@app.route('/<path:filename>')
def serve_static(filename):
    print(f"DEBUG: Serving static file: {filename} from {public_folder}", file=sys.stderr)
    return send_from_directory(public_folder, filename)

@app.route('/predict', methods=['POST'])
def predict():
    print("DEBUG: /predict endpoint hit.", file=sys.stderr)
    try:
        # Uploads directory at the project root level on Render
        uploads_dir = os.path.join(project_root_on_render, 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        print(f"DEBUG: Uploads directory: {uploads_dir}", file=sys.stderr)
        
        if not os.path.exists(model_path) or not os.path.exists(classes_path):
            print(f"ERROR: Model or classes files missing. Model exists: {os.path.exists(model_path)}, Classes exists: {os.path.exists(classes_path)}", file=sys.stderr)
            return jsonify({'error': 'Required model files are missing'}), 500

        query = request.form.get('query')

        if not query:
            return jsonify({'error': 'No query provided'}), 400
            
        pdf = request.files.get('pdf')
        xray = request.files.get('xray')
        eye_image = request.files.get('eyeImage')

        if pdf and pdf != 'no_image_data':
            pdf.save(os.path.join(uploads_dir, pdf.filename))
            print(f"DEBUG: Saved PDF: {pdf.filename}", file=sys.stderr)
            
        if xray and xray != 'no_image_data':
            xray.save(os.path.join(uploads_dir, xray.filename))
            print(f"DEBUG: Saved Xray: {xray.filename}", file=sys.stderr)
            
        eye_disease_prediction = 'N/A'
        if eye_image and eye_image != 'no_image_data':
            print("DEBUG: Eye image received, processing for prediction.", file=sys.stderr)
            try:
                upload_file_path = os.path.join(uploads_dir, eye_image.filename)
                eye_image.save(upload_file_path)
                if not os.path.exists(upload_file_path):
                    raise FileNotFoundError(f"Failed to save image: {upload_file_path}")
                print(f"DEBUG: Eye image saved to: {upload_file_path}", file=sys.stderr)
            except Exception as e:
                print(f"ERROR: Error saving eye image: {e}", file=sys.stderr)
                return jsonify({'error': f'Error processing eye image: {str(e)}'}), 500

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"DEBUG: Using device: {device}", file=sys.stderr)

            with open(classes_path, "r") as f:
                classes = json.load(f)
            print(f"DEBUG: Classes loaded: {classes}", file=sys.stderr)
                
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, len(classes))

            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            print("DEBUG: Model loaded and set to eval mode.", file=sys.stderr)

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

            eye_img_for_prediction = upload_file_path          
            eye_disease_prediction = predict_image(eye_img_for_prediction)
            print(f"DEBUG: Prediction for {eye_img_for_prediction}: {eye_disease_prediction}", file=sys.stderr)

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
        print("DEBUG: Sending request to Hugging Face model...", file=sys.stderr)
        completion = client.chat.completions.create(
            model="Intelligent-Internet/II-Medical-8B-1706",
            messages=[{"role": "user", "content": structured_prompt}],
        )
        print("DEBUG: Hugging Face model response received.", file=sys.stderr)

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
        print(f"ERROR: Exception in /predict endpoint: {e}", file=sys.stderr)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("DEBUG: Running app in debug mode locally.", file=sys.stderr)
    app.run(debug=True)
