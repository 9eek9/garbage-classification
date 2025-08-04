import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import gradio as gr
import matplotlib.pyplot as plt
from transformers import BlipProcessor, BlipForConditionalGeneration

# Import predict function
from predict import predict_image

# ---------------------------
# 1. Load ResNet50 model (state_dict)
# ---------------------------
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(classes))  # 6 classes
state_dict = torch.load("../models/best_garbage_model.pth", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Transform for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------------------------
# 2. Setup LLM - BLIP 
# ---------------------------

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def verify_with_blip(image, predicted_label, confidence, threshold=0.50):
    """
    Generate a friendly explanation for the predicted label using BLIP only.
    Adds a warning if confidence is below threshold.
    """
    # --- BLIP Caption ---
    try:
        inputs = blip_processor(image, return_tensors="pt").to(device)
        out = blip_model.generate(**inputs)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)

        if confidence < (threshold * 100):  
            return (
                f"<span style='color:#FF6B6B;'>⚠️ Low Confidence ({confidence:.2f}%). "
                f"This image looks like '{caption}', but "
                f"the model predicts '{predicted_label}', so the result may be unreliable.</span>"
            )

        return (
            f"LLM suggests: This image looks like '{caption}', "
            f"so classifying it as '{predicted_label}' seems reasonable!"
        )
    except Exception as e:
        print(f"[Error] BLIP captioning failed: {e}")
        return "Friendly explanation unavailable at the moment."


# ---------------------------
# 3. Visualization: Donut Chart
# ---------------------------
def create_donut_chart(top3):
    labels = [item[0] for item in top3]
    sizes = [item[1] for item in top3]
    colors = ['#66b3ff', '#99ff99', '#ffcc99']

    fig, ax = plt.subplots(figsize=(4, 4))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        colors=colors,
        startangle=140
    )
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)
    ax.axis('equal')
    plt.tight_layout()
    return fig

# ---------------------------
# 4. Gradio prediction function
# ---------------------------
def gradio_predict(image):
    # Step 1: Get top-3 predictions
    pred_label, pred_conf, top3 = predict_image(image)

    # Step 2: Create Donut Chart for top-3
    donut_fig = create_donut_chart(top3)

    # Step 3: Friendly explanation (using BLIP)
    verification = verify_with_blip(image.convert("RGB"), pred_label, pred_conf)

    # Step 4: Style HTML for output
    friendly_html = f"""
        <div style='background-color:#3A506B; padding:15px; border-radius:10px;
                    border:1px solid #E0E0E0; color:#F1F1F1; font-size:16px;'>
            <b>Model Prediction:</b> {pred_label} ({pred_conf:.2f}%)<br><br>
            {verification}
        </div>
        """

    return donut_fig, pred_label, friendly_html

# ---------------------------
# 5. Custom CSS for Gradio
# ---------------------------
custom_css = """
body, .gradio-container {
    background-color: #2C2F33 !important;
}
"""

# ---------------------------
# 6. Launch Gradio app
# ---------------------------
iface = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Plot(label="Top-3 Donut Chart"),
        gr.Textbox(label="Model predicted class", lines=1),
        gr.HTML(label="Friendly LLM Explanation")
    ],
    title="Garbage Classification",
    description="Upload an image to classify the waste type and receive a friendly explanation.",
    css=custom_css
)

if __name__ == "__main__":
    iface.launch()
