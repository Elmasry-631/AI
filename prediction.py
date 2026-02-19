import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
import gradio as gr

from learning_utils import fine_tune_from_dataset, save_labeled_image

MODEL_PATH = "best_model.pth"


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_names


model = None
class_names = []

try:
    model, class_names = load_model()
except FileNotFoundError:
    pass


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def predict(img: Image.Image):
    if model is None or not class_names:
        return {"error": 1.0}

    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    return {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}


def teach_model(img: Image.Image, label: str):
    global model, class_names

    if img is None or not label or not label.strip():
        return "Please provide both image and label."

    saved_path = save_labeled_image(img, label)
    try:
        stats = fine_tune_from_dataset(MODEL_PATH, data_dir="data", epochs=1)
    except FileNotFoundError as exc:
        return f"Saved: {saved_path}\n{exc}"

    model, class_names = load_model()

    return (
        f"Saved: {saved_path}\n"
        f"Updated classes: {', '.join(stats['classes'])}\n"
        f"Total samples: {stats['samples']} | Last loss: {stats['last_loss']:.4f}"
    )


with gr.Blocks(title="Self-Learning Image Classifier") as iface:
    gr.Markdown("# Self-Learning Image Classifier")
    gr.Markdown("ارفع صورة للتصنيف، ولو التسمية غلط ارفعها مرة تانية مع التصنيف الصحيح علشان النموذج يتعلم منها.")

    with gr.Tab("Predict"):
        pred_img = gr.Image(type="pil")
        pred_output = gr.Label(num_top_classes=5)
        pred_btn = gr.Button("Predict")
        pred_btn.click(fn=predict, inputs=pred_img, outputs=pred_output)

    with gr.Tab("Teach"):
        teach_img = gr.Image(type="pil", label="Image")
        teach_label = gr.Textbox(label="Correct label")
        teach_output = gr.Textbox(label="Training log")
        teach_btn = gr.Button("Save & Learn")
        teach_btn.click(fn=teach_model, inputs=[teach_img, teach_label], outputs=teach_output)

iface.launch()
