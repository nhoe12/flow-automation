import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import joblib
import time
from imblearn.over_sampling import SMOTE

print("Đang nạp dữ liệu flow_dataset_merged.csv...")
df = pd.read_csv("flow_dataset_merged.csv", low_memory=False)

X = df.drop(columns=["Label"])
y = df["Label"]

# Phân chia Train/Test giữ nguyên tỉ lệ lớp gốc
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("\n--- Kích thước tập Train TRƯỚC khi dùng SMOTE ---")
print(y_train.value_counts())

print("\nĐang áp dụng SMOTE nhắm mục tiêu riêng cho lớp ARP_MITM_LAB...")
# CHỈ tăng ARP_MITM_LAB lên 20.000 mẫu, bỏ qua hoàn toàn các lớp khác (như Heartbleed)
sampling_strategy = {"ARP_MITM_LAB": 20000} 
smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=5)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("\n--- Kích thước tập Train SAU khi dùng SMOTE ---")
print(y_train.value_counts())

print("\nBắt đầu huấn luyện RandomForestClassifier...")
start = time.time()
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
elapsed = time.time() - start

print("\nĐang đánh giá mô hình trên tập Test (dữ liệu thật chưa qua tổng hợp)...")
y_pred = model.predict(X_test)

print(f"\nThời gian huấn luyện: {elapsed:.2f} giây")
print(f"Precision (weighted): {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print(f"Recall (weighted): {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print(f"F1 (weighted): {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")

print("\nChi tiết theo từng lớp:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nKết quả riêng cho lớp mục tiêu của đề tài - ARP_MITM_LAB:")
print(classification_report(y_test, y_pred, labels=["ARP_MITM_LAB"], zero_division=0))

joblib.dump(model, "anomaly_model_flow.joblib")
print("\nĐã lưu mô hình thành công: anomaly_model_flow.joblib")