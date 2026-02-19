import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import gradio as gr

# ==========================================
# 1. Load Checkpoint and Metadata
# ==========================================
# Load the dictionary containing weights and class names
checkpoint = torch.load("best_model.pth", map_location="cpu")

# Extract dynamic data saved during training
class_names = checkpoint['class_names']
num_classes = checkpoint['num_classes']

# ==========================================
# 2. Reconstruct Model Architecture
# ==========================================
model = models.resnet18(weights=None)
# Adjust the final fully connected layer to match the saved classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load the weights into the model
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ==========================================
# 3. Prediction Logic
# ==========================================
def predict(img: Image.Image):
    # Standard ResNet preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Transform and add batch dimension
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_t)
        # Calculate probabilities using Softmax
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        # Create a dictionary of {ClassName: Probability} for Gradio Label output
        return {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}

# ==========================================
# 4. Gradio Interface
# ==========================================
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=num_classes),
    title="Dynamic Image Classifier",
    description=f"Model trained to recognize: {', '.join(class_names)}"
)

iface.launch()
