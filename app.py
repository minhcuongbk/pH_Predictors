from flask import Flask, request, render_template
import pickle
import numpy as np

# Load mô hình ML
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

# Danh sách các loại dầu từ file Excel (37 loại)
oil_names = [
    'BACH HO', 'BONNY LT', 'Heavy Bach Ho', 'YELLOW TUNA', 'KIMANIS', 'AMNA', 'QUA IBOE',
    'MURBAN', 'MIRI LT', 'BUNGA ORKID', 'KIKEH', 'CABINDA', 'RABI BLEND', 'RABI LIGHT',
    'BERTAM', 'CHAMPION', 'DAI HUNG', 'LABUAN', 'RUBY', 'TE GIAC TRANG', 'AZERI LT',
    "N'KOSSA", 'RANG DONG', 'CHIM SAO', 'THANG LONG', 'SOKOL', 'HAI THACH', 'SU TU DEN',
    'ESPO', 'WTI', 'SONG DOC', 'BU ATTIFEL', 'MINAS', 'PALANCA', 'FORCADOS',
    'SLOPS', 'CDU RESIDUE'
]

@app.route('/')
def index():
    return render_template('index.html', oil_names=oil_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        inputs = [float(request.form.get(oil, 0)) for oil in oil_names]
        X = np.array(inputs).reshape(1, -1)
        predicted_ph = model.predict(X)[0]
        return render_template('index.html', prediction=round(predicted_ph, 2), oil_names=oil_names)
    except Exception as e:
        return render_template('index.html', prediction=f"Lỗi: {e}", oil_names=oil_names)

if __name__ == '__main__':
    app.run(debug=True)
