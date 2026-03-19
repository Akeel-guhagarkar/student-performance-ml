from fastapi import FastAPI
import joblib

app = FastAPI()

# load trained model
model = joblib.load("model/student_model.pkl")

@app.get("/")
def home():
    return {"message": "Student Performance Prediction API"}

@app.post("/predict")
def predict(data: dict):

    values = [list(data.values())]

    prediction = model.predict(values)

    return {"prediction": int(prediction[0])}from fastapi import FastAPI
import joblib

app = FastAPI()

# load trained model
model = joblib.load("model/student_model.pkl")

@app.get("/")
def home():
    return {"message": "Student Performance Prediction API"}

@app.post("/predict")
def predict(data: dict):

    values = [list(data.values())]

    prediction = model.predict(values)

    return {"prediction": int(prediction[0])}