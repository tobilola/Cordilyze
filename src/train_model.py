import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb
import pickle
import warnings
import os
warnings.filterwarnings('ignore')

# Generate synthetic CVD dataset for demo
np.random.seed(42)

def generate_synthetic_data(n_samples=2000):
    """Generate realistic CVD dataset"""
    data = {
        'age': np.random.randint(30, 80, n_samples),
        'sex': np.random.choice([0, 1], n_samples),
        'cholesterol_total': np.random.normal(200, 40, n_samples),
        'cholesterol_hdl': np.random.normal(50, 15, n_samples),
        'cholesterol_ldl': np.random.normal(120, 35, n_samples),
        'triglycerides': np.random.normal(150, 50, n_samples),
        'blood_pressure_systolic': np.random.normal(130, 20, n_samples),
        'blood_pressure_diastolic': np.random.normal(80, 12, n_samples),
        'glucose': np.random.normal(100, 25, n_samples),
        'bmi': np.random.normal(27, 5, n_samples),
        'smoking': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'physical_activity': np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.5, 0.2]),
    }
    
    df = pd.DataFrame(data)
    
    # Create target based on clinical risk factors
    risk_score = (
        (df['age'] - 30) * 0.5 +
        df['sex'] * 10 +
        (df['cholesterol_total'] - 200) * 0.3 +
        (df['cholesterol_ldl'] - 100) * 0.4 +
        (df['blood_pressure_systolic'] - 120) * 0.5 +
        (df['glucose'] - 100) * 0.3 +
        (df['bmi'] - 25) * 2 +
        df['smoking'] * 15 -
        df['physical_activity'] * 5 +
        np.random.normal(0, 10, n_samples)
    )
    
    df['cvd_risk'] = (risk_score > np.median(risk_score)).astype(int)
    
    return df

print("Generating synthetic CVD dataset...")
df = generate_synthetic_data(2000)

# Split features and target
X = df.drop('cvd_risk', axis=1)
y = df['cvd_risk']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# NOTE: No StandardScaler needed — tree-based models are scale-invariant.
# Random Forest, XGBoost, and Gradient Boosting split on feature values
# directly, so scaling doesn't affect their decisions. Removing the scaler
# also means the app can feed raw patient values straight into predict_proba()
# without needing to load and apply a scaler first.

print("\nTraining ensemble models...")

# Model 1: Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

# Model 2: XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

# Model 3: Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_proba = gb.predict_proba(X_test)[:, 1]

# Ensemble predictions (weighted voting)
ensemble_proba = 0.35 * rf_proba + 0.40 * xgb_proba + 0.25 * gb_proba
ensemble_pred = (ensemble_proba > 0.5).astype(int)

# Evaluate
print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"\nRandom Forest:")
print(f"  Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
print(f"  ROC-AUC:  {roc_auc_score(y_test, rf_proba):.4f}")

print(f"\nXGBoost:")
print(f"  Accuracy: {accuracy_score(y_test, xgb_pred):.4f}")
print(f"  ROC-AUC:  {roc_auc_score(y_test, xgb_proba):.4f}")

print(f"\nGradient Boosting:")
print(f"  Accuracy: {accuracy_score(y_test, gb_pred):.4f}")
print(f"  ROC-AUC:  {roc_auc_score(y_test, gb_proba):.4f}")

print(f"\nENSEMBLE MODEL:")
print(f"  Accuracy: {accuracy_score(y_test, ensemble_pred):.4f}")
print(f"  ROC-AUC:  {roc_auc_score(y_test, ensemble_proba):.4f}")

print("\n" + classification_report(y_test, ensemble_pred))

# Create models directory
if not os.path.exists('models'):
    os.makedirs('models')

# Save models
print("\nSaving models...")
with open('models/rf_model.pkl', 'wb') as f:
    pickle.dump(rf, f)
with open('models/xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
with open('models/gb_model.pkl', 'wb') as f:
    pickle.dump(gb, f)

# Save feature names
with open('models/feature_names.txt', 'w') as f:
    f.write('\n'.join(X.columns.tolist()))

print("Models saved successfully!")
print("\nNext step: Run the Streamlit app with: streamlit run cordilyze_app.py")
