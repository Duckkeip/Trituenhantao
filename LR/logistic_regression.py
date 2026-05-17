import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Tải và tiền xử lý dữ liệu
df = pd.read_csv('framingham.csv')

# Xử lý các giá trị còn thiếu bằng cách loại bỏ (hoặc điền khuyết nếu muốn)
df_clean = df.dropna()

# Tách đặc trưng (X) và nhãn mục tiêu (y)
X = df_clean.drop(columns=['TenYearCHD'])
y = df_clean['TenYearCHD']

# 2. Chia tập dữ liệu thành tập Huấn luyện (Train) và tập Kiểm tra (Test) theo tỷ lệ 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Chuẩn hóa dữ liệu (Feature Scaling) vì Logistic Regression nhạy cảm với thang đo
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- THÀNH PHẦN 1: MÔ HÌNH LOGISTIC REGRESSION MẶC ĐỊNH ---
model_default = LogisticRegression(random_state=42, max_iter=1000)
model_default.fit(X_train_scaled, y_train)
y_pred_def = model_default.predict(X_test_scaled)

# --- THÀNH PHẦN 2: MÔ HÌNH CÂN BẰNG TRỌNG SỐ (BALANCED CLASS WEIGHTS) ---
model_balanced = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
model_balanced.fit(X_train_scaled, y_train)
y_pred_bal = model_balanced.predict(X_test_scaled)

# 4. Hàm hiển thị kết quả đánh giá
def evaluate_model(y_true, y_pred, name):
    print(f"=== KẾT QUẢ ĐÁNH GIÁ: {name} ===")
    print(f"Accuracy (Độ chính xác tổng thể): {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision (Độ chính xác dự đoán dương tính): {precision_score(y_true, y_pred):.4f}")
    print(f"Recall (Độ nhạy / Tỷ lệ tìm thấy bệnh): {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score (Điểm F1 trung bình điều hòa): {f1_score(y_true, y_pred):.4f}")
    print("Confusion Matrix (Ma trận nhầm lẫn):")
    print(confusion_matrix(y_true, y_pred))
    print("\n")

# In kết quả của cả 2 mô hình ra màn hình
evaluate_model(y_test, y_pred_def, "Mô hình Mặc định (Default)")
evaluate_model(y_test, y_pred_bal, "Mô hình Cân bằng trọng số (Balanced)")

# 5. Vẽ và lưu biểu đồ Confusion Matrix cho mô hình Mặc định
sns.heatmap(confusion_matrix(y_test, y_pred_def), annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Không bị bệnh (0)', 'Bị bệnh (1)'], 
            yticklabels=['Không bị bệnh (0)', 'Bị bệnh (1)'])
plt.xlabel('Nhãn Dự Đoán (Predicted)')
plt.ylabel('Nhãn Thực Tế (True)')
plt.title('Confusion Matrix - Logistic Regression (Default)')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()