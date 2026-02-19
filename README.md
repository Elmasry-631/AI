# AI Image Classifier

## الخوارزميات المستخدمة حاليًا
- **Transfer Learning** باستخدام `ResNet18` مدرب مسبقًا على ImageNet (الإعداد الافتراضي في التدريب).
- **Custom CNN** بسيط (3 بلوكات Convolution + BatchNorm + ReLU + Pooling) كبديل عند إلغاء `USE_TRANSFER`.
- **Optimizer**: Adam.
- **Loss Function**: CrossEntropyLoss.
- **Learning Rate Scheduler**: ReduceLROnPlateau.
- **Early Stopping** عبر `PATIENCE`.

## الجديد: التعلم من تصحيحك للتصنيف
تم إضافة مسار "Teach" داخل واجهة Gradio:
1. ترفع الصورة.
2. تكتب التصنيف الصحيح.
3. النظام يحفظها داخل `data/<label>/`.
4. يعمل **fine-tuning سريع** على آخر طبقة ويحدث `best_model.pth`.

## الجديد: التعلم من الإنترنت
تم إضافة سكربت `ingest_from_web.py`:
1. أنشئ ملف `web_sources.json` بناءً على `web_sources.example.json`.
2. أضف روابط صور متاحة لكل فئة.
3. شغّل:

```bash
python ingest_from_web.py
```

السكربت سينزل الصور داخل `data/` ثم يعمل fine-tuning.

## ملاحظات مهمة
- "التعلم من كل حاجة متاحة" بشكل مفتوح غير آمن وقد يسبب بيانات سيئة أو مخالفة حقوق/تراخيص.
- الأفضل تعمل **curation** لمصادر الإنترنت (روابط موثوقة) قبل التدريب.
- يفضل إعادة تدريب كامل دوريًا (`train_script.py`) بعد زيادة كبيرة في البيانات.
