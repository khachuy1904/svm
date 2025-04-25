import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt

# ===== 1. Đọc dữ liệu =====
df = pd.read_csv("online_shoppers_intention.csv")

# ===== 2. Encode nhãn =====
df['Revenue'] = df['Revenue'].map({True: 1, False: -1})

# ===== 3. One-hot encoding =====
df = pd.get_dummies(df, columns=['Month', 'VisitorType', 'Weekend'], drop_first=True)

# ===== 4. Xử lý missing value =====
df = df.dropna()

# ===== 5. Tách đặc trưng và nhãn =====
X = df.drop('Revenue', axis=1).values
y = df['Revenue'].values.astype(float)

# ===== 6. Train/Test split =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===== 7. Chuẩn hóa =====
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===== 8. RBF Kernel =====
def rbf_kernel(x1, x2, sigma=1.0):
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * sigma ** 2))

# ===== 9. Tính ma trận Gram (train) =====
n_samples = X_train.shape[0]
K = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    for j in range(n_samples):
        K[i, j] = rbf_kernel(X_train[i], X_train[j])

# ===== 10. QP input =====
P = matrix(np.outer(y_train, y_train) * K)
q = matrix(-np.ones(n_samples))
G = matrix(-np.eye(n_samples))
h = matrix(np.zeros(n_samples))
A = matrix(y_train, (1, n_samples), 'd')
b = matrix(0.0)

# ===== 11. Solve QP =====
solvers.options['show_progress'] = False
sol = solvers.qp(P, q, G, h, A, b)
alphas = np.ravel(sol['x'])

# ===== 12. Support vectors =====
threshold = 1e-5
sv = alphas > threshold
sv_X = X_train[sv]
sv_y = y_train[sv]
sv_alpha = alphas[sv]

# ===== 13. Dự đoán =====
def project(x):
    result = 0
    for a, sv_yi, sv_xi in zip(sv_alpha, sv_y, sv_X):
        result += a * sv_yi * rbf_kernel(x, sv_xi)
    return result

def predict(X_test):
    return np.sign([project(x) for x in X_test])

# ===== 14. Dự đoán và đánh giá =====
y_pred = predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label=1)
rec = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)

print(f"\n🎯 Evaluation on Test Set:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")

# ===== 15. Confusion Matrix =====
cm = confusion_matrix(y_test, y_pred, labels=[1, -1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Revenue=True", "Revenue=False"])
disp.plot()
plt.title("Confusion Matrix - SVM with RBF Kernel")
plt.show()
