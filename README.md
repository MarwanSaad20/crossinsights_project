# CrossInsights Project

**CrossInsights** هو مشروع Python متكامل لمعالجة البيانات، التحليل الاستكشافي، بناء نظام توصية متقدم، **وتنبؤ تقييمات المستخدمين**. يجمع المشروع بين تقنيات التوصية الذكية (SVD وKNN) ونماذج التنبؤ الآلي (Linear Regression وRandom Forest) لتقديم منصة شاملة لتحليل سلوك المستخدمين والتنبؤ بتفضيلاتهم.

---

## الميزات الرئيسية

✨ **نظام توصية ذكي**: باستخدام SVD وKNN (user-user وitem-item)
🔮 **تنبؤ تقييمات المستخدمين**: باستخدام Linear Regression وRandom Forest
📊 **تحليل استكشافي متقدم**: رسوم بيانية تفاعلية وإحصائيات شاملة
⚙️ **pipeline تلقائي**: تشغيل جميع العمليات بأمر واحد
🧪 **اختبارات شاملة**: Unit tests لضمان جودة الكود
📈 **تحليل مقارن للنماذج**: مقارنة أداء نماذج التنبؤ المختلفة

---

# الهيكل العام للمشروع

```
C:/crossinsights_project
│
├── crossinsights/                    # الحزمة الأساسية للمشروع
│   ├── __init__.py
│   ├── data/                        # ملفات البيانات (raw, processed)
│   │   ├── raw/                     # البيانات الأصلية كما تم تحميلها
│   │   │   ├── users.csv
│   │   │   ├── movies.csv
│   │   │   └── ratings.csv
│   │   └── processed/               # البيانات بعد تنظيفها وتحويلها
│   │       ├── users_clean.csv
│   │       ├── movies_clean.csv
│   │       ├── ratings_clean.csv
│   │       ├── knn_recommendations.csv
│   │       └── predicted_ratings.csv    # 🆕 تنبؤات النماذج
│   │
│   ├── utils/                       # دوال مساعدة
│   │   ├── __init__.py
│   │   ├── data_loader.py          # تحميل البيانات
│   │   ├── eda_tools.py            # دوال التحليل الاستكشافي
│   │   ├── preprocessing.py        # تنظيف وتحويل البيانات
│   │   ├── knn_recommender.py      # دوال توصية KNN
│   │   └── predictors_utils.py     # 🆕 دوال مساعدة للتنبؤ
│   │
│   ├── analysis/                   # التحليلات والاستكشاف
│   │   ├── __init__.py
│   │   ├── eda.ipynb              # Notebook تحليلي
│   │   ├── rating_distribution.png
│   │   ├── genre_popularity.png
│   │   ├── age_vs_ratings.png
│   │   ├── knn_analysis.png       # توزيع التشابه
│   │   └── predictors_analysis.png # 🆕 تحليل نماذج التنبؤ
│   │
│   ├── models/                     # نماذج التوصية والتنبؤ
│   │   ├── __init__.py
│   │   ├── placeholder.py         # نموذج SVD
│   │   ├── inspect_model.py       # فحص نموذج SVD
│   │   ├── svd_model.pkl
│   │   ├── knn_model.pkl
│   │   ├── predictors.py          # 🆕 كود تدريب نماذج التنبؤ
│   │   ├── linear_regression_model.pkl  # 🆕 نموذج الانحدار الخطي
│   │   └── random_forest_model.pkl      # 🆕 نموذج Random Forest
│   │
│   ├── config/
│   │   └── config.yaml            # إعدادات المسارات والنماذج
│   │
│   └── run_pipeline.py            # ملف تشغيل كامل الـ pipeline
│
├── tests/                         # اختبارات الوحدة
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_knn_recommender.py
│   └── test_predictors.py         # 🆕 اختبارات نماذج التنبؤ
│
├── notebooks/                     # Notebooks تجريبية
│
├── requirements.txt               # مكتبات المشروع
├── environment.yml               # بيئة Conda (اختياري)
└── README.md                     # وثائق المشروع
```

---

## المتطلبات

- Python 3.10 أو أحدث
- مكتبات المشروع موجودة في `requirements.txt`، يمكن تثبيتها عبر:

```bash
pip install -r requirements.txt
```

أو باستخدام Conda:
```bash
conda env create -f environment.yml
conda activate crossinsights
```

### المكتبات المطلوبة
```
pandas
numpy
matplotlib
seaborn
scikit-learn      # لدعم KNN، نماذج التنبؤ وcosine similarity
joblib
pyyaml
```

---

## تشغيل الـ Pipeline

يمكنك تشغيل جميع مراحل المشروع (تحميل البيانات، تنظيفها، التحليل، التوصيات، **والتنبؤ**) باستخدام:

```bash
python -m crossinsights.run_pipeline
```

### العمليات التي تتم تلقائياً:

1. **📥 تحميل البيانات**: من مجلد `data/raw/` (users.csv, movies.csv, ratings.csv)
2. **🧹 تنظيف البيانات**: معالجة القيم المفقودة والمكررة وحفظها في `data/processed/`
3. **📊 التحليل الاستكشافي**: إنشاء الرسوم البيانية في `analysis/`
4. **🤝 توصيات KNN**: توليد التوصيات وحفظها في `knn_recommendations.csv`
5. **🎯 توصيات SVD**: تدريب النموذج وتوليد التوصيات
6. **🔮 تدريب نماذج التنبؤ**: Linear Regression و Random Forest
7. **📈 توليد التنبؤات**: حفظ النتائج في `predicted_ratings.csv`
8. **📋 تحليل الأداء**: مقارنة النماذج ورسوم الأداء

---

## الوظائف الجديدة: نماذج التنبؤ 🆕

### 🎯 الهدف
تنبؤ تقييم المستخدم لفيلم معين بناءً على:
- **خصائص المستخدم**: العمر، الجنس، المهنة
- **خصائص الفيلم**: النوع، سنة الإصدار
- **أنماط التقييم**: التاريخ التقييمي للمستخدم

### 📊 النماذج المستخدمة
1. **Linear Regression**: للعلاقات الخطية البسيطة
2. **Random Forest**: للعلاقات المعقدة والتفاعلات بين المتغيرات

### 📄 المخرجات الجديدة
- **`predicted_ratings.csv`**: يحتوي على:
  ```
  userId, movieId, predicted_rating_linear, predicted_rating_forest, actual_rating
  ```
- **`predictors_analysis.png`**: رسم مقارن بين:
  - توزيع التقييمات الحقيقية
  - توزيع تنبؤات Linear Regression
  - توزيع تنبؤات Random Forest

### 🧪 كيفية عمل التنبؤ

1. **دمج البيانات**: يتم دمج جداول Users، Movies، وRatings في جدول واحد
2. **إعداد المتغيرات**:
   - **X (المدخلات)**: خصائص المستخدم + خصائص الفيلم
   - **y (الهدف)**: التقييم الفعلي
3. **التدريب**: تدريب النماذج على 80% من البيانات
4. **التنبؤ**: اختبار على 20% المتبقية
5. **الحفظ**: حفظ النماذج والتنبؤات

---

## الاختبارات

تشغيل جميع الاختبارات:
```bash
python -m pytest tests/
```

أو اختبار نماذج التنبؤ فقط:
```bash
python -m pytest tests/test_predictors.py
```

### الاختبارات المتوفرة:
- ✅ **test_data_loader.py**: اختبار تحميل البيانات
- ✅ **test_preprocessing.py**: اختبار تنظيف البيانات
- ✅ **test_knn_recommender.py**: اختبار توصيات KNN
- 🆕 **test_predictors.py**: اختبار نماذج التنبؤ

---

## الاستخدام المتقدم

### 1. الحصول على توصيات لمستخدم معين:
```python
from crossinsights.utils.knn_recommender import recommend_user_user
recommendations = recommend_user_user(user_id=1, top_n=10)
print(recommendations)
```

### 2. تنبؤ تقييم مستخدم لفيلم: 🆕
```python
from crossinsights.models.predictors import predict_rating
predicted_score = predict_rating(user_id=1, movie_id=150)
print(f"التقييم المتوقع: {predicted_score}")
```

### 3. تحليل أداء النماذج: 🆕
```python
from crossinsights.utils.predictors_utils import evaluate_models
metrics = evaluate_models()
print("مقاييس الأداء:", metrics)
```

---

## التطوير المستقبلي

### 📋 المخطط له:
- 🔄 **نماذج هجينة**: دمج التوصيات والتنبؤ في نموذج واحد
- 🎛️ **تحسين المعاملات**: Hyperparameter tuning للنماذج
- 🌐 **واجهة ويب**: تطبيق Flask/Django لسهولة الاستخدام
- 📱 **API**: REST API للتكامل مع التطبيقات الأخرى
- 🧠 **Deep Learning**: استخدام Neural Networks للتنبؤ
- 💾 **قاعدة بيانات**: دعم PostgreSQL/MongoDB

### 🔧 التحسينات الممكنة:
- **Weighted KNN**: إعطاء أوزان مختلفة للجيران
- **Matrix Factorization**: تقنيات أكثر تطوراً من SVD
- **Content-based filtering**: توصيات بناء على محتوى الأفلام
- **Ensemble Methods**: دمج عدة نماذج تنبؤ

---

## المساهمة

### 👨‍💻 المطورون:
**Marwan Al_Jubouri** – تطوير وتصميم:
- 🏗️ الهيكل العام للمشروع
- 🧹 معالجة وتنظيف البيانات
- 📊 التحليل الاستكشافي
- 🤖 نماذج التوصية (SVD وKNN)
- 🔮 نماذج التنبؤ (Linear Regression وRandom Forest)
- 🧪 اختبارات الوحدة الشاملة

### 🤝 للمساهمة:
1. Fork المشروع
2. إنشاء branch جديد (`git checkout -b feature/amazing-feature`)
3. Commit التغييرات (`git commit -m 'Add amazing feature'`)
4. Push للـ branch (`git push origin feature/amazing-feature`)
5. فتح Pull Request

---

## الملاحظات المهمة

### 📁 **ملفات الإخراج**:
جميع الملفات الناتجة يتم حفظها تلقائياً:
- `processed/`: البيانات المعالجة والنتائج
- `analysis/`: الرسوم البيانية والتحليلات
- `models/`: النماذج المدربة (.pkl files)

### 📝 **السجلات (Logging)**:
- سجلات مفصلة لكل عملية في الـ console
- تتبع الأخطاء والتحذيرات
- معلومات الأداء والوقت المستغرق

### 🔒 **الأمان**:
- التعامل مع مشكلة البداية الباردة (Cold-start problem)
- معالجة البيانات المفقودة والقيم الشاذة
- validation للمدخلات والمخرجات

### 🚀 **الأداء**:
- معالجة فعالة للبيانات الكبيرة
- تحسين الذاكرة واستخدام CPU
- حفظ النماذج لإعادة الاستخدام السريع

---

## رخصة المشروع

هذا المشروع مطور للأغراض التعليمية والبحثية. يُرجى احترام حقوق الملكية الفكرية عند الاستخدام.

---

## الدعم والمساعدة

للمساعدة أو الإبلاغ عن مشاكل:
- 📧 **البريد الإلكتروني**: [البريد الخاص بالمطور]
- 🐛 **Issues**: استخدم GitHub Issues للإبلاغ عن الأخطاء
- 💬 **المناقشة**: GitHub Discussions للأسئلة العامة

---

**🎯 CrossInsights - حيث تلتقي التوصيات الذكية بالتنبؤ الدقيق!**
