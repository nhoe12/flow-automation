import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import joblib
import time
from imblearn.over_sampling import SMOTE

# 1. Đọc dữ liệu
print("Đang nạp dữ liệu...")
df = pd.read_csv("flow_dataset_merged.csv", low_memory=False)

X = df.drop(columns=["Label"])
y = df["Label"]

# 2. Chia tập Train/Test (70/30) giữ nguyên tỉ lệ lớp
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("\n--- Kích thước tập Train TRƯỚC khi dùng SMOTE ---")
print(y_train.value_counts())

# 3. Xử lý mất cân bằng lớp bằng SMOTE (Chỉ trên tập Train)
print("\nĐang áp dụng SMOTE để tổng hợp dữ liệu lớp thiểu số...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("\n--- Kích thước tập Train SAU khi dùng SMOTE ---")
print(y_train.value_counts())

# 4. Huấn luyện mô hình Random Forest
print("\nĐang huấn luyện mô hình Random Forest...")
start = time.time()
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
elapsed = time.time() - start

# 5. Đánh giá mô hình trên tập Test (dữ liệu chưa qua tổng hợp)
y_pred = model.predict(X_test)

print(f"\nThời gian huấn luyện: {elapsed:.2f} giây")
print(f"Precision (weighted): {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall (weighted): {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1 (weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")

print("\nChi tiết theo từng lớp:")
print(classification_report(y_test, y_pred))

# Riêng lớp ARP_MITM_LAB
print("\nKết quả riêng cho lớp ARP_MITM_LAB:")
print(classification_report(y_test, y_pred, labels=["ARP_MITM_LAB"]))

# 6. Lưu mô hình
joblib.dump(model, "anomaly_model_flow.joblib")
print("Đã lưu mô hình thành công: anomaly_model_flow.joblib")