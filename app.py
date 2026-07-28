from flask import Flask, render_template, request, jsonify
from engine import get_predictor

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON body received'}), 400

        p = get_predictor()
        predicted_age, years_left = p.predict(data)
        factors  = p.factor_impacts(data)
        health   = p.health_score(data)

        return jsonify({
            'predicted_age': predicted_age,
            'years_left':    years_left,
            'current_age':   float(data['age']),
            'factors':       factors,
            'health_score':  health,
        })
    except KeyError as e:
        return jsonify({'error': f'Missing field: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
