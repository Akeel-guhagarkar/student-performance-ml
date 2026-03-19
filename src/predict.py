import joblib

# load model
model = joblib.load("model/student_model.pkl")

def predict_student(data):

    prediction = model.predict([data])

    if prediction[0] == 1:
        return "Student will PASS"
    else:
        return "Student is AT RISK"


# example test
sample_student = [3.2,80,75,70,72,8,4,1,30]

result = predict_student(sample_student)

print(result)