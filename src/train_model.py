import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from preprocess import load_data, split_features_target

# load dataset
df = load_data()

# split features and target
X, y = split_features_target(df)

# split train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# create model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# train model
model.fit(X_train, y_train)

# predictions
predictions = model.predict(X_test)

# evaluation
print(classification_report(y_test, predictions))

# save model
joblib.dump(model, "model/student_model.pkl")

print("Model trained and saved successfully!")