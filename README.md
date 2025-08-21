CrossInsights Project

CrossInsights هو مشروع Python متكامل لمعالجة البيانات، التحليل الاستكشافي، بناء نظام توصية متقدم، وتنبؤ تقييمات المستخدمين. يجمع المشروع بين تقنيات التوصية الذكية (SVD وKNN) ونماذج التنبؤ الآلي (Linear Regression وRandom Forest) لتقديم منصة شاملة لتحليل سلوك المستخدمين والتنبؤ بتفضيلاتهم.

الآن أصبح المشروع عمليًا أكثر عبر واجهة CLI للتوصيات (🆕 المهمة 10)، حيث يستطيع المستخدم تجربة النظام مباشرة بإدخال اسم فيلم والحصول على توصيات مشابهة.

الميزات الرئيسية

✨ نظام توصية ذكي: باستخدام SVD وKNN (user-user وitem-item)
🔮 تنبؤ تقييمات المستخدمين: باستخدام Linear Regression وRandom Forest
📊 تحليل استكشافي متقدم: رسوم بيانية تفاعلية وإحصائيات شاملة
⚙️ pipeline تلقائي: تشغيل جميع العمليات بأمر واحد
🧪 اختبارات شاملة: Unit tests لضمان جودة الكود
📈 تحليل مقارن للنماذج: مقارنة أداء نماذج التنبؤ المختلفة
💻 واجهة CLI للتوصيات 🆕: إدخال فيلم والحصول على توصيات مباشرة

الهيكل العام للمشروع
C:/crossinsights_project
│
├── crossinsights/                    # الحزمة الأساسية للمشروع
│   ├── __init__.py
│   ├── data/                        # ملفات البيانات (raw, processed)
│   │   ├── raw/
│   │   │   ├── users.csv
│   │   │   ├── movies.csv
│   │   │   └── ratings.csv
│   │   └── processed/
│   │       ├── users_clean.csv
│   │       ├── movies_clean.csv
│   │       ├── ratings_clean.csv
│   │       ├── knn_recommendations.csv
│   │       └── predicted_ratings.csv
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── eda_tools.py
│   │   ├── preprocessing.py
│   │   ├── knn_recommender.py
│   │   └── predictors_utils.py
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── eda.ipynb
│   │   ├── rating_distribution.png
│   │   ├── genre_popularity.png
│   │   ├── age_vs_ratings.png
│   │   ├── knn_analysis.png
│   │   └── predictors_analysis.png
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── placeholder.py
│   │   ├── inspect_model.py
│   │   ├── svd_model.pkl
│   │   ├── knn_model.pkl
│   │   ├── predictors.py
│   │   ├── linear_regression_model.pkl
│   │   └── random_forest_model.pkl
│   │
│   ├── config/
│   │   └── config.yaml
│   │
│   ├── run_pipeline.py               # pipeline متكامل
│   └── cli_recommender.py            # 🆕 واجهة CLI للتوصيات
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_knn_recommender.py
│   └── test_predictors.py
│
├── notebooks/
├── requirements.txt
├── environment.yml
└── README.md

تشغيل الـ Pipeline

لتشغيل جميع مراحل المشروع دفعة واحدة:

python -m crossinsights.run_pipeline

🆕 تشغيل واجهة CLI للتوصيات

الآن يمكنك تجربة التوصيات بشكل مباشر:

cd C:/crossinsights_project
python -m crossinsights.cli_recommender

مثال عملي:
🎬 أهلاً بك في CrossInsights Recommender!

📂 جاري تحميل البيانات والنماذج...

🎥 أدخل اسم فيلم تحبه: Pulp Fiction

✅ تم العثور على الفيلم: Pulp Fiction (1994)

📌 أفضل 5 توصيات (KNN):
1. Inglourious Basterds (2009)
2. Platoon (1986)
...

📌 أفضل 5 توصيات (SVD):
1. The Handmaiden (2016)
2. The Green Mile (1999)
...

كيف تعمل؟

تحميل البيانات النظيفة (movies_clean.csv, ratings_clean.csv)

تحميل النماذج المدربة (KNN, SVD)

البحث عن الفيلم وإيجاد التطابق الأقرب

استدعاء دوال التوصية (KNN وSVD)

عرض أفضل N توصيات (افتراضيًا: 5)

العلاقة بالمشروع

نفس البيانات: نستخدم الملفات المنظفة داخل processed/

نفس النماذج: KNN وSVD المدربة مسبقًا

نفس الـ pipeline: الواجهة مجرد طبقة إضافية لسهولة الاستخدام

التطوير المستقبلي للـ CLI 📝

إضافة وضع Notebook لعرض التوصيات برسوم وصور

خيار Predict: إدخال (UserId, MovieId) للحصول على التقييم المتوقع

دعم فلترة بالأنواع/السنة لزيادة دقة النتائج
