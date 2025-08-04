import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# Class names
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# ---------------------------
# 1. Recreate the model
# ---------------------------
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))  # 6 classes

# Load weights
state_dict = torch.load("../models/best_garbage_model.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# ---------------------------
# 2. Transform for single image
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(img, topk=3):
    # Convert and preprocess
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    else:
        img = img.convert("RGB")
    
    img_t = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_t)
        probs = F.softmax(outputs, dim=1)[0]

        # Top-K predictions
        topk_prob, topk_idx = torch.topk(probs, topk)
        topk_labels = [(class_names[idx], float(prob) * 100) 
                       for idx, prob in zip(topk_idx, topk_prob)]
        
        # Final top-1 prediction
        pred_label, pred_conf = topk_labels[0]
    
    return pred_label, pred_conf, topk_labels
