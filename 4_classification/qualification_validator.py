from joblib import dump, load


classification_model = load("models/qualification_by_double_grade_model.joblib")

while True:
    grades = [int(g) for g in input("Enter space separated Technical grade and English grade:\n").split()]
    X = [grades]
    model_response = classification_model.predict(X)[0]
    model_confidence = classification_model.predict_proba(X)[0][1]
    if model_response == 0:
        text_response = "fails"
    elif model_response == 1:
        text_response = "passes"
    response_confidence = int(round(abs(200 * model_confidence - 100)))
    print("The given candidate {0} with the {1}% confidence.".format(text_response, response_confidence))
