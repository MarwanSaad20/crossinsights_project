# CrossInsights Project

**CrossInsights** هو مشروع Python متكامل لمعالجة البيانات، التحليل الاستكشافي، وبناء نظام توصية باستخدام نموذجي SVD وKNN (user-user وitem-item). يهدف المشروع إلى تمكين المستخدمين من الحصول على توصيات أفلام ذكية بناءً على بيانات المستخدمين والتقييمات.

---

# الهيكل العام للمشروع

C:/crossinsights_project
│
├── crossinsights/ # الحزمة الأساسية للمشروع
│ ├── init.py
│ ├── data/ # ملفات البيانات (raw, processed)
│ │ ├── raw/ # البيانات الأصلية كما تم تحميلها
│ │ │ ├── users.csv
│ │ │ ├── movies.csv
│ │ │ └── ratings.csv
│ │ └── processed/ # البيانات بعد تنظيفها وتحويلها
│ │ ├── users_clean.csv
│ │ ├── movies_clean.csv
│ │ ├── ratings_clean.csv
│ │ └── knn_recommendations.csv  # جديد
│ │
│ ├── utils/ # دوال مساعدة (تحميل، تنظيف، تحليل، توصيات)
│ │ ├── init.py
│ │ ├── data_loader.py # تحميل البيانات
│ │ ├── eda_tools.py # دوال التحليل الاستكشافي
│ │ ├── preprocessing.py # تنظيف وتحويل البيانات
│ │ └── knn_recommender.py # دوال توصية KNN (جديد)
│ │
│ ├── analysis/ # التحليلات والاستكشاف
│ │ ├── init.py
│ │ ├── eda.ipynb # Notebook تحليلي
│ │ ├── rating_distribution.png
│ │ ├── genre_popularity.png
│ │ ├── age_vs_ratings.png
│ │ └── knn_analysis.png # توزيع التشابه (جديد)
│ │
│ ├── models/ # نماذج التوصية/التنبؤ
│ │–├── init.py
│ │ ├── placeholder.py # نموذج SVD
│ │ ├── inspect_model.py # فحص نموذج SVD
│ │ ├── svd_model.pkl
│ │ └── knn_model.pkl # نموذج KNN (جديد)
│ │
│ ├── config/
│ │ └── config.yaml # إعدادات المسارات، إعدادات النماذج
│ │
│ └── run_pipeline.py # ملف لتشغيل كامل الـ pipeline تلقائيًا
│
├── tests/ # اختبارات الوحدة
│ ├── init.py
│ ├── test_data_loader.py
│ ├── test_preprocessing.py
│ └── test_knn_recommender.py # اختبارات KNN (جديد)
│
├── notebooks/ # Notebooks تجريبية أو تحليل إضافي
│
├── requirements.txt # مكتبات المشروع
├── environment.yml # بيئة Conda (اختياري)
└── README.md # وثائق المشروع


---

## المتطلبات

- Python 3.10 أو أحدث
- مكتبات المشروع موجودة في `requirements.txt`، يمكن تثبيتها عبر:

```bash
pip install -r requirements.txt

أو باستخدام Conda:bash

conda env create -f environment.yml
conda activate crossinsights

المكتبات المطلوبةpandas
numpy
matplotlib
seaborn
scikit-learn  # جديد لدعم KNN وcosine similarity
joblib
pyyaml

تشغيل الـ Pipelineيمكنك تشغيل جميع مراحل المشروع (تحميل البيانات، تنظيفها، التحليل، توصيات KNN، تدريب SVD، وتوليد التوصيات) باستخدام:bash

python -m crossinsights.run_pipeline

ستتم العمليات التالية تلقائيًا:تحميل البيانات الأصلية (users.csv, movies.csv, ratings.csv) من مجلد data/raw.
تنظيف وتحويل البيانات وحفظها في data/processed.
إجراء التحليل الاستكشافي وإنشاء الرسوم البيانية في analysis/.
توليد توصيات KNN وحفظها في data/processed/knn_recommendations.csv وإنشاء رسم توزيع التشابه في analysis/knn_analysis.png.
تدريب نموذج SVD وحفظه في models/.
توليد توصيات SVD وعرض النتائج في الـ console.

التوصيات والتطوير المستقبلييمكن إضافة دعم لتوليد توصيات لمستخدمين متعددين دفعة واحدة.
تحسين تحليل البيانات الاستكشافي باستخدام تقنيات إضافية مثل matplotlib، seaborn، أو plotly.
تطوير نموذج توصية أكثر تطورًا باستخدام LightFM أو Implicit Feedback Models.
توسيع KNN ليشمل Weighted KNN أو Hybrid approach.

المساهمونMarwan Al_Jubouri – تطوير الهيكل، تنظيف البيانات، النماذج (SVD وKNN)، والتحليل الاستكشافي.

الملاحظاتجميع الملفات الناتجة من المعالجة والتحليل يتم حفظها تلقائيًا في مجلدات processed/ وanalysis/.
سجلات التشغيل (logs) موجودة في crossinsights/logs/ لتتبع أي خطأ أو معلومات مفيدة عن التنفيذ.

#### 6. تحديث: `requirements.txt`
إضافة `scikit-learn` إلى قائمة المكتبات المطلوبة.

pandas
numpy
matplotlib
seaborn
scikit-learn  # جديد
joblib
pyyaml

#### 7. ملاحظات التكامل
- **حفظ المسارات**: تم الالتزام بالمسارات المحددة في `config.yaml` لضمان التناسق مع الهيكلية الحالية.
- **الحفاظ على الوظائف الحالية**: لم يتم تعديل أي وظائف تتعلق بتحميل البيانات، التنظيف، التحليل الاستكشافي، أو نموذج SVD، مما يضمن استقرار النظام.
- **التعامل مع مشكلة البداية الباردة (Cold-start)**: تم التعامل معها في وظائف KNN عبر إرجاع DataFrame فارغة مع رسالة تحذير في السجل.
- **الجودة والاحترافية**: الكود منظم، موثق، ومدعوم باختبارات وسجلات تفصيلية لضمان الموثوقية.
- **التوسع المستقبلي**: تصميم KNN مرن لدعم إضافات مثل Weighted KNN أو Hybrid approach.

---

### الخطوات التالية
1. **التثبيت**: قم بتحديث المكتبات باستخدام:
   ```bash
   pip install -r requirements.txt

التشغيل: شغّل الـ pipeline بالأمر:bash

python -m crossinsights.run_pipeline

