import joblib
import pandas as pd

# تحميل النموذج
model_path = "C:/crossinsights_project/crossinsights/models/svd_model.pkl"
svd_model = joblib.load(model_path)

# معلومات أساسية
print("==== معلومات عامة عن النموذج ====")
print("عدد المكونات (n_components):", svd_model.n_components)
print("عدد التكرارات (n_iter):", svd_model.n_iter)
print("عدد المستخدمين/الأفلام في المصفوفة:", svd_model.components_.shape)

# حفظ المكونات في ملف CSV لتصفحها بسهولة
components_csv = model_path.replace(".pkl", "_components.csv")
pd.DataFrame(svd_model.components_).to_csv(components_csv, index=False)
print(f"تم حفظ المكونات في CSV: {components_csv}")

# يمكنك إضافة أي تفاصيل إضافية هنا، مثل استعراض أول 10 عناصر
print("\n==== أول 10 عناصر من المكونات ====")
print(svd_model.components_[:, :10])
