# Directly Run the CODE to visit my webpage...
from flask import Flask, render_template_string, request
import threading, webbrowser, os, joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

app = Flask(__name__)

MODEL_PATH = "model.pkl"

# TRAIN MODEL IF NOT FOUND
if not os.path.exists(MODEL_PATH):
    print("⚙️ Training a small demo model (first time only)...")
    np.random.seed(42)
    X = pd.DataFrame({
        "Pregnancies": np.random.randint(0, 10, 500),
        "Glucose": np.random.randint(80, 200, 500),
        "BloodPressure": np.random.randint(50, 120, 500),
        "BMI": np.random.uniform(15, 45, 500),
        "Age": np.random.randint(18, 80, 500)
    })
    y = (X["Glucose"] > 125).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    pipe = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=100, random_state=42))])
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, MODEL_PATH)
    print("✅ Model trained & saved as model.pkl")

# LOAD MODEL
model = joblib.load(MODEL_PATH)

#  HTML TEMPLATE
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Diabetes Prediction</title>
<script src="https://cdn.jsdelivr.net/npm/particles.js@2.0.0/particles.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap" rel="stylesheet">
<style>
body {
  margin:0; font-family:'Poppins',sans-serif;
  background:linear-gradient(135deg,#74ABE2,#5563DE);
  height:100vh; overflow:hidden; display:flex; justify-content:center; align-items:center;
}
.card {
  background:white; padding:40px 60px; border-radius:25px;
  box-shadow:0 20px 40px rgba(0,0,0,0.2);
  text-align:center; position:relative; animation:fadeIn 1s ease;
}
h1 { margin-bottom:10px; color:#333; font-size:28px; }
.dev { font-size:14px; color:#666; margin-bottom:25px; font-weight:500; }
input[type=number] {
  width:100%; padding:10px; border:1px solid #ccc; border-radius:10px;
  margin-bottom:8px; font-size:16px;
}
.suggestion {
  font-size:13px; color:#666; margin-bottom:12px;
  background:#f8f8f8; padding:4px 8px; border-radius:6px; transition:0.3s;
}
button {
  background:#5563DE; color:white; border:none; padding:12px 30px;
  border-radius:30px; cursor:pointer; font-size:16px; transition:0.3s;
}
button:hover { background:#3542b5; transform:scale(1.05); }
.result {
  margin-top:25px; font-size:22px; font-weight:600;
  padding:15px; border-radius:12px; animation:slideIn 0.8s ease;
}
.result.good { background:#D4EDDA; color:#155724; }
.result.bad { background:#F8D7DA; color:#721C24; }
@keyframes fadeIn { from {opacity:0; transform:translateY(20px);} to {opacity:1; transform:translateY(0);} }
@keyframes slideIn { from {opacity:0; transform:translateY(10px);} to {opacity:1; transform:translateY(0);} }
#particles-js { position:fixed; top:0; left:0; width:100%; height:100%; z-index:-1; }
</style>
</head>
<body>
<div id="particles-js"></div>
<div class="card">
  <h1>🤖 AI Diabetes Prediction</h1>
  <div class="dev">Developer: S. Lavan Chary</div>

  <form method="POST">
    <input type="number" name="pregnancies" placeholder="Pregnancies" required oninput="showSuggestion(this)">
    <div class="suggestion" id="s_preg"></div>

    <input type="number" name="glucose" placeholder="Glucose" required oninput="showSuggestion(this)">
    <div class="suggestion" id="s_glucose"></div>

    <input type="number" name="bloodpressure" placeholder="Blood Pressure" required oninput="showSuggestion(this)">
    <div class="suggestion" id="s_bp"></div>

    <input type="number" step="0.1" name="bmi" placeholder="BMI" required oninput="showSuggestion(this)">
    <div class="suggestion" id="s_bmi"></div>

    <input type="number" name="age" placeholder="Age" required oninput="showSuggestion(this)">
    <div class="suggestion" id="s_age"></div>

    <button type="submit">Predict</button>
  </form>

  {% if result %}
    <div class="result {{ 'bad' if result == 'Diabetic' else 'good' }}">
      Prediction: {{ result }} <br>
      Confidence: {{ confidence }}%
    </div>
  {% endif %}
</div>

<script>
particlesJS("particles-js", {
  "particles":{"number":{"value":90},"size":{"value":3},"move":{"speed":1},"line_linked":{"enable":true}},
  "interactivity":{"events":{"onhover":{"enable":true,"mode":"repulse"}}}
});

// Show live health suggestions
function showSuggestion(input){
  let val = parseFloat(input.value);
  let id = input.name;
  if(isNaN(val)) return;
  let sug="";
  if(id=="glucose"){
    sug = val>125 ? "⚠️ High glucose – risk of diabetes." : "✅ Normal glucose (below 125).";
    document.getElementById("s_glucose").innerText = sug;
  }
  else if(id=="bmi"){
    sug = val>25 ? "⚠️ Overweight range – try to keep BMI < 25." : "✅ Healthy BMI range.";
    document.getElementById("s_bmi").innerText = sug;
  }
  else if(id=="bloodpressure"){
    sug = val>120 ? "⚠️ High BP, watch diet & stress." : "✅ Normal BP range.";
    document.getElementById("s_bp").innerText = sug;
  }
  else if(id=="age"){
    sug = val>50 ? "⚠️ Age risk increases after 50." : "✅ Low age risk.";
    document.getElementById("s_age").innerText = sug;
  }
  else if(id=="pregnancies"){
    sug = val>6 ? "⚠️ More pregnancies may increase risk." : "✅ Normal range.";
    document.getElementById("s_preg").innerText = sug;
  }
}
</script>
</body>
</html>
"""

# ROUTE
@app.route("/", methods=["GET", "POST"])
def home():
    result = confidence = None
    if request.method == "POST":
        vals = [float(request.form[k]) for k in ["pregnancies","glucose","bloodpressure","bmi","age"]]
        pred_prob = model.predict_proba([vals])[0][1]
        result = "Diabetic" if pred_prob >= 0.5 else "Non-Diabetic"
        confidence = round(pred_prob*100 if result=="Diabetic" else (1-pred_prob)*100, 2)
    return render_template_string(HTML_TEMPLATE, result=result, confidence=confidence)

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == "__main__":
    threading.Timer(1, open_browser).start()
    app.run(debug=False)

