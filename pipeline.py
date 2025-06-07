import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('433457ff-3c5d-4d9f-a797-1f4588a538bf.csv')

# Drop irrelevant columns
df = df.drop(columns=['show_id', 'title', 'description'])

# Set target column
target_column = 'type'
X = df.drop(columns=[target_column])
y = df[target_column]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Identify feature types
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Pipelines for preprocessing
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Complete pipeline with model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
accuracy = pipeline.score(X_test, y_test)
print(f"\n✅ Model Accuracy: {accuracy:.2%}")

# Save the pipeline
joblib.dump(pipeline, 'netflix_model_pipeline.pkl')
print("🧠 Model pipeline saved as 'netflix_model_pipeline.pkl'")

# --- Optional: Feature Importance ---
# Extract feature names
cat_columns = pipeline.named_steps['preprocessor'].transformers_[1][1] \
    .named_steps['encoder'].get_feature_names_out(categorical_features)
all_features = numeric_features + cat_columns.tolist()

importances = pipeline.named_steps['classifier'].feature_importances_
importance_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Show top 10 features
print("\n📊 Top 10 Important Features:")
print(importance_df.head(10))

# Plot
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'][:10], importance_df['Importance'][:10])
plt.xlabel("Importance Score")
plt.title("Top 10 Important Features")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
