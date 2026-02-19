import torch
from torchvision import models, transforms
from PIL import Image
import requests

# 1. تحميل الموديل الجاهز كاملاً (ResNet18)
# الموديل ده متدرب على 1000 نوع من الأشياء
device = torch.device("cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# 2. تحميل أسماء الفئات (Labels) 
# الموديل بيطلع رقم، وإحنا محتاجين نعرف الرقم ده معناه إيه (مثلاً رقم 248 = Beagle)
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(LABELS_URL).text.splitlines()

# 3. تجهيز الصورة
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_t)
            
        # الحصول على أعلى احتمالية
        _, index = torch.max(outputs, 1)
        percentage = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        
        result = labels[index[0]]
        confidence = percentage[index[0]].item()
        
        print(f"\nالتوقع: {result}")
        print(f"نسبة التأكد: {confidence:.2f}%")
        
    except Exception as e:
        print(f"حدث خطأ: {e}")

# 4. طلب المسار من المستخدم
path = input("أدخل مسار الصورة اللي عاوز تعرف هي إيه: ").strip()
predict_image(path)
