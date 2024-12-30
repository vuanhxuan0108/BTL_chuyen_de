from flask import Flask, render_template, request
import joblib
from preprocess import TextPreprocessor
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        content = request.form['content']
        if content.strip() == "":
            return render_template('index.html', error = "Trường này không được để trống")
        else:
            model_classifier = joblib.load('SVM_best_model.pkl')
            tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
            label_encoder = joblib.load('label_encoder.joblib')
            preprocessor = TextPreprocessor(r"vietnamese-stopwords.txt")
            sample_text_processed = preprocessor.preprocess_text(content)

            sample_text_transformed = tfidf_vectorizer.transform([sample_text_processed])

            predicted_label_encoded = model_classifier.predict(sample_text_transformed)

            predicted_label = label_encoder.inverse_transform(predicted_label_encoded)
            return render_template('index.html', prediction=predicted_label[0])

if __name__ == '__main__':
    app.run(debug=True)
