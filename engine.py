import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

SMOKE_MAP    = {'never': 0, 'former': 1, 'light': 2, 'heavy': 3}
ALCOHOL_MAP  = {'none': 0, 'moderate': 1, 'heavy': 2}
EXERCISE_MAP = {'none': 0, 'light': 1, 'moderate': 2, 'intense': 3}
DIET_MAP     = {'poor': 0, 'average': 1, 'good': 2, 'excellent': 3}
STRESS_MAP   = {'low': 0, 'moderate': 1, 'high': 2, 'very_high': 3}
SLEEP_MAP    = {'less_5': 0, '5_6': 1, '7_8': 2, 'more_9': 3}
GENDER_MAP   = {'male': 0, 'female': 1, 'other': 0.5}


def _encode(data):
    return [
        float(data['age']),
        GENDER_MAP.get(data['gender'], 0.5),
        float(data['bmi']),
        SMOKE_MAP.get(data['smoking'], 0),
        ALCOHOL_MAP.get(data['alcohol'], 0),
        EXERCISE_MAP.get(data['exercise'], 0),
        DIET_MAP.get(data['diet'], 1),
        STRESS_MAP.get(data['stress'], 1),
        SLEEP_MAP.get(data['sleep'], 2),
        float(data['systolic_bp']),
        float(data['cholesterol']),
        float(data['blood_sugar']),
        int(data.get('diabetes', 0)),
        int(data.get('heart_disease', 0)),
        int(data.get('hypertension', 0)),
        int(data.get('family_longevity', 0)),
        float(data.get('income_level', 2)),
        float(data.get('education_years', 12)),
    ]


def _generate_training_data(n=3000):
    np.random.seed(99)
    age        = np.random.uniform(18, 80, n)
    gender     = np.random.choice([0, 0.5, 1], n, p=[0.49, 0.02, 0.49])
    bmi        = np.random.normal(26, 5, n).clip(15, 50)
    smoking    = np.random.choice([0,1,2,3], n, p=[0.5, 0.2, 0.2, 0.1])
    alcohol    = np.random.choice([0,1,2], n, p=[0.4, 0.45, 0.15])
    exercise   = np.random.choice([0,1,2,3], n, p=[0.25, 0.30, 0.30, 0.15])
    diet       = np.random.choice([0,1,2,3], n, p=[0.20, 0.35, 0.30, 0.15])
    stress     = np.random.choice([0,1,2,3], n, p=[0.20, 0.35, 0.30, 0.15])
    sleep      = np.random.choice([0,1,2,3], n, p=[0.10, 0.25, 0.50, 0.15])
    sys_bp     = np.random.normal(125, 20, n).clip(80, 200)
    chol       = np.random.normal(200, 40, n).clip(100, 350)
    sugar      = np.random.normal(95, 20, n).clip(60, 300)
    diabetes   = (sugar > 126).astype(float) * np.random.binomial(1, 0.7, n)
    heart_dis  = np.random.binomial(1, 0.08, n)
    hyper      = (sys_bp > 140).astype(float) * np.random.binomial(1, 0.75, n)
    fam_long   = np.random.binomial(1, 0.35, n)
    income     = np.random.uniform(1, 5, n)
    edu_years  = np.random.uniform(8, 22, n)

    life = (
        82.0
        + (gender - 0.5) * 4
        - np.abs(bmi - 22) * 0.35
        - smoking * 3.5
        - alcohol * 2.0
        + exercise * 2.5
        + diet * 2.0
        - stress * 1.8
        + (sleep - 1) * 1.2
        - (sys_bp - 120) * 0.08
        - (chol - 180) * 0.025
        - (sugar - 90) * 0.04
        - diabetes * 5.0
        - heart_dis * 7.0
        - hyper * 3.5
        + fam_long * 3.0
        + income * 0.8
        + edu_years * 0.15
        + np.random.normal(0, 2.5, n)
    ).clip(45, 105)

    X = np.column_stack([
        age, gender, bmi, smoking, alcohol, exercise, diet,
        stress, sleep, sys_bp, chol, sugar, diabetes, heart_dis,
        hyper, fam_long, income, edu_years
    ])
    return X, life


class LifeExpectancyPredictor:
    def __init__(self):
        X, y = _generate_training_data()
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        self.model = GradientBoostingRegressor(
            n_estimators=300, max_depth=5,
            learning_rate=0.07, subsample=0.8, random_state=42
        )
        self.model.fit(Xs, y)

    def predict(self, data):
        vec = np.array([_encode(data)])
        vec_s = self.scaler.transform(vec)
        raw = float(self.model.predict(vec_s)[0])
        age = float(data['age'])
        return round(raw, 1), round(max(0, raw - age), 1)

    def factor_impacts(self, data):
        base_s = self.scaler.transform([_encode(data)])[0]
        base   = float(self.model.predict([base_s])[0])
        tweaks = {
            'smoking':     ({'smoking': 'never'},    'Stop Smoking',           'lifestyle'),
            'exercise':    ({'exercise': 'intense'},  'Exercise Regularly',    'lifestyle'),
            'diet':        ({'diet': 'excellent'},    'Improve Diet',          'lifestyle'),
            'alcohol':     ({'alcohol': 'none'},      'Reduce Alcohol',        'lifestyle'),
            'stress':      ({'stress': 'low'},        'Manage Stress',         'mental'),
            'sleep':       ({'sleep': '7_8'},         'Optimise Sleep',        'mental'),
            'bmi':         ({'bmi': 22.5},            'Reach Healthy BMI',     'physical'),
            'sys_bp':      ({'systolic_bp': 115},     'Control Blood Pressure','physical'),
            'cholesterol': ({'cholesterol': 170},     'Lower Cholesterol',     'physical'),
            'blood_sugar': ({'blood_sugar': 85},      'Control Blood Sugar',   'physical'),
        }
        factors = []
        for _, (override, label, category) in tweaks.items():
            mod = {**data, **override}
            ms  = self.scaler.transform([_encode(mod)])[0]
            mp  = float(self.model.predict([ms])[0])
            delta = round(mp - base, 1)
            if delta > 0.1:
                factors.append({'label': label, 'delta': delta, 'category': category})
        factors.sort(key=lambda x: -x['delta'])
        return factors[:6]

    def health_score(self, data):
        ls = 0
        ls += {'never':3,'former':2,'light':1,'heavy':0}.get(data['smoking'], 0) * 6
        ls += {'none':3,'moderate':2,'heavy':0}.get(data['alcohol'], 0) * 4
        ls += {'none':0,'light':1,'moderate':2,'intense':3}.get(data['exercise'], 0) * 5
        ls += {'poor':0,'average':1,'good':2,'excellent':3}.get(data['diet'], 0) * 4

        bmi   = float(data['bmi'])
        bp    = float(data['systolic_bp'])
        chol  = float(data['cholesterol'])
        sugar = float(data['blood_sugar'])
        ph = (
            max(0, 10 - abs(bmi - 22) * 1.2) +
            max(0, 10 - max(0, bp - 120) * 0.25) +
            max(0, 10 - max(0, chol - 180) * 0.08) +
            max(0, 10 - max(0, sugar - 90) * 0.15)
        )

        ms = 0
        ms += {'low':3,'moderate':2,'high':1,'very_high':0}.get(data['stress'], 0) * 3.5
        ms += {'less_5':0,'5_6':1,'7_8':3,'more_9':2}.get(data['sleep'], 0) * 2.5

        total = round((ls + ph + ms) / 100 * 100)
        return {
            'total': min(100, total),
            'breakdown': {
                'Lifestyle': round(ls / 40 * 100),
                'Physical':  round(ph / 40 * 100),
                'Mental':    round(ms / 20 * 100),
            }
        }


_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = LifeExpectancyPredictor()
    return _predictor
