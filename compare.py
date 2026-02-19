import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
import os

# 1. استخدام موديل ResNet18 جاهز (Pre-trained)
# الموديل ده "خبير" وجاهز لاستخراج الميزات فوراً
device = torch.device("cpu")
# بنشيل آخر طبقة (اللي بتصنف الصور) عشان إحنا عاوزين الـ Features بس
model = nn.Sequential(*list(model.children())[:-1])
model.eval()

# 2. التحويلات (لازم نستخدم معايير ResNet القياسية)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_embedding(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(img)
        return embedding.flatten() # تحويل الميزات لشكل مبسط
    except Exception as e:
        print(f"Error: {e}")
        return None

# 3. التفاعل مع المستخدم
print("\n--- نظام المقارنة الاحترافي (ResNet) ---")
img1_path = input("أدخل مسار الصورة الأولى: ").strip()
img2_path = input("أدخل مسار الصورة الثانية: ").strip()

if os.path.exists(img1_path) and os.path.exists(img2_path):
    emb1 = get_embedding(img1_path)
    emb2 = get_embedding(img2_path)

    if emb1 is not None and emb2 is not None:
        # حساب التشابه
        cos_sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
        percentage = cos_sim.item() * 100
        
        print("-" * 30)
        print(f"نسبة التشابه الحقيقية: {percentage:.2f}%")
        print("-" * 30)
        
        if percentage > 85:
            print("النتيجة: الصور متشابهة جداً")
        else:
            print("النتيجة: الصور مختلفة")
else:
    print("المسارات غير صحيحة!")
