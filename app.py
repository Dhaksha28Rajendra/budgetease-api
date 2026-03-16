# app.py
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model  # type: ignore
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def p(name: str) -> str:
    return os.path.join(BASE_DIR, name)


app = Flask(__name__)

# -----------------------------
# Load models / scalers
# -----------------------------
# 1) Profile model
model = joblib.load(p("budgetease_multi_model.pkl"))
scaler = joblib.load(p("scaler.pkl"))
le_spender = joblib.load(p("spender_encoder.pkl"))
le_income = joblib.load(p("income_encoder.pkl"))

# 2) Expense prediction model
expense_model = load_model(p("expense_lstm_model.keras"))
scaler_X_pred = joblib.load(p("scaler_X.pkl"))

# 3) Real-time budget adjustment model
budget_adjust_model = joblib.load(p("supervised_rf_budget_model.pkl"))
budget_scaler = joblib.load(p("final_budget_scaler.pkl"))

# 4) Recommendation model
reco_model = joblib.load(p("budget_recommendation_rf_model.pkl"))
reco_le_income = joblib.load(p("le_income.pkl"))
reco_le_spender = joblib.load(p("le_spender.pkl"))

print("Income encoder classes:", getattr(reco_le_income, "classes_", None))
print("Spender encoder classes:", getattr(reco_le_spender, "classes_", None))
print("Reco model expects:", getattr(reco_model, "n_features_in_", None))
print("Reco feature names:", getattr(reco_model, "feature_names_in_", None))

# -----------------------------
# Helpers
# -----------------------------
income_range_mapping = {
    1: 8000,
    2: 15000,
    3: 25000,
    4: 35000
}


def encode_first(values):
    if values is None:
        return 0.0
    if isinstance(values, (list, tuple)):
        if len(values) == 0:
            return 0.0
        v = values[0]
    else:
        v = values
    try:
        return float(int(v) - 1)
    except Exception:
        try:
            return float(v)
        except Exception:
            return 0.0


def safe_pct(amount: float, income: float) -> float:
    if income <= 0:
        return 0.0
    return (amount / income) * 100.0


def map_income_label(raw: str) -> str:
    s = (raw or "").strip().lower()
    if "low" in s:
        return "Low"
    if "moderate" in s or "medium" in s:
        return "Medium"
    if "high" in s:
        return "High"
    # already clean?
    if raw in getattr(reco_le_income, "classes_", []):
        return raw
    return "Low"


def map_spender_label(raw: str) -> str:
    s = (raw or "").strip().lower()
    if "saver" in s:
        return "Saver"
    if "balanced" in s:
        if "Balanced" in getattr(reco_le_spender, "classes_", []):
            return "Balanced"
        if "Balanced Spender" in getattr(reco_le_spender, "classes_", []):
            return "Balanced Spender"
        return "Balanced"
    if "impulsive" in s:
        return "Impulsive"
    # already matches?
    if raw in getattr(reco_le_spender, "classes_", []):
        return raw
    return "Balanced"


def encode_with_labelencoder(le, value: str, default: int = 0) -> int:
    try:
        arr = le.transform([value])
        return int(arr[0])
    except Exception:
        try:
            classes = getattr(le, "classes_", [])
            for i, c in enumerate(classes):
                if str(c).strip().lower() == str(value).strip().lower():
                    return i
        except Exception:
            pass
        return default


# -----------------------------
# 1) Profile prediction (rule based)
# -----------------------------
@app.route("/predict_profile", methods=["POST"])
def predict_profile():
    data = request.get_json(force=True)
    print("PROFILE INPUT:", data)

    avg_income_choice = int(data.get("avg_income_choice", 1))
    spending_choice = int(data.get("approx_spending_choice", 1))

    # -------------------------
    # Income classification
    # -------------------------
    if avg_income_choice in [1, 2]:
        income_type = "low"
    elif avg_income_choice == 3:
        income_type = "medium"
    else:
        income_type = "high"

    # -------------------------
    # Spender classification
    # -------------------------
    if spending_choice in [1, 2]:
        spender_type = "saver"
    elif spending_choice == 3:
        spender_type = "balanced"
    else:
        spender_type = "impulsive"

    return jsonify({
        "spender_type": spender_type,
        "income_type": income_type
    })


# -----------------------------
# 2) Expense percentage prediction
# -----------------------------
@app.route("/predict_expense", methods=["POST"])
def predict_expense():
    data = request.get_json(force=True)

    try:
        features = [
            float(data.get("income_mean", 0)),
            float(data.get("expense_mean", 0)),
            float(data.get("essentials_expense", 0)),
            float(data.get("academic_expense", 0)),
            float(data.get("leisure_expense", 0)),
            float(data.get("other_expense", 0)),
            float(data.get("expense_ratio", 0)),
        ]

        X = np.array([features], dtype=np.float32)

        X_scaled = scaler_X_pred.transform(X)
        X_seq = X_scaled.reshape(1, 1, X_scaled.shape[1])

        y_pred = expense_model.predict(X_seq, verbose=0)
        pred = y_pred[0].tolist()

        pred_pct = [round(v * 100, 2) for v in pred]

        return jsonify({
            "essentials_pct": pred_pct[0],
            "academic_pct": pred_pct[1],
            "leisure_pct": pred_pct[2],
            "other_pct": pred_pct[3],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# 3) Real-time budget adjustment
# -----------------------------
@app.route("/adjust_budget", methods=["POST"])
def adjust_budget():
    data = request.get_json(force=True)

    age = float(data.get("age", 0))
    spender_type = str(data.get("spender_type", "")).strip()
    income_type = str(data.get("income_type", "")).strip()
    total_income = float(data.get("total_income", 0))

    e = float(data.get("essentials_pct", 0))
    a = float(data.get("academic_pct", 0))
    l = float(data.get("leisure_pct", 0))
    o = float(data.get("other_pct", 0))

    # Convert categorical labels to integers the scaler/model expects
    spender_enc = encode_with_labelencoder(le_spender, spender_type, default=0)
    income_enc = encode_with_labelencoder(le_income, income_type, default=0)

    # Build numeric array (note dtype=float)
    X = np.array([[age, spender_enc, income_enc, total_income, e, a, l, o]], dtype=float)

    try:
        # scale then predict
        X_scaled = budget_scaler.transform(X)
        pred = budget_adjust_model.predict(X_scaled)
        pred = np.array(pred).reshape(-1)
        ess_amt, aca_amt, lei_amt, oth_amt = map(float, pred[:4])

    except Exception as ex:
        print("Budget adjust model error:", ex)
        ess_amt = total_income * (e / 100.0)
        aca_amt = total_income * (a / 100.0)
        lei_amt = total_income * (l / 100.0)
        oth_amt = total_income * (o / 100.0)

    # keep totals consistent (small rounding fix)
    diff = total_income - (ess_amt + aca_amt + lei_amt + oth_amt)
    oth_amt += diff

    return jsonify({
        "essentials_amount": round(ess_amt, 2),
        "academic_amount": round(aca_amt, 2),
        "leisure_amount": round(lei_amt, 2),
        "other_amount": round(oth_amt, 2),
    })


# -----------------------------
# 4) Recommendation model (dynamic rule-based messages)
# -----------------------------
@app.route("/recommend_budget", methods=["POST"])
def recommend_budget():
    data = request.get_json(force=True)

    monthly_income = float(data.get("monthly_income", 0))

    essentials_amt = float(data.get("essentials_amount", 0))
    academic_amt = float(data.get("academic_amount", 0))
    leisure_amt = float(data.get("leisure_amount", 0))
    other_amt = float(data.get("other_amount", 0))

    income_type_raw = str(data.get("income_type", "")).strip()
    spender_type_raw = str(data.get("spender_type", "")).strip()

    if monthly_income <= 0:
        return jsonify({"error": "monthly_income must be > 0"}), 400

    # Convert to percentages
    e_pct = (essentials_amt / monthly_income) * 100.0
    a_pct = (academic_amt / monthly_income) * 100.0
    l_pct = (leisure_amt / monthly_income) * 100.0
    o_pct = (other_amt / monthly_income) * 100.0

    total_spend_pct = e_pct + a_pct + l_pct + o_pct

    # -------------------------
    # Dynamic STATUS detection
    # -------------------------
    if total_spend_pct > 95:
        status = "Overspending"
    elif total_spend_pct < 60:
        status = "Saving"
    else:
        status = "Balanced"

    # -------------------------
    # Dynamic recommendations
    # -------------------------
    recommendations = []

    if a_pct > 30:
        recommendations.append(
            "Academic spending is relatively high. Consider planning education purchases and applying for discounts."
        )

    if l_pct > 20:
        recommendations.append(
            "Leisure spending is rising — try setting a weekly entertainment cap."
        )

    if o_pct > 15:
        recommendations.append(
            "Your miscellaneous spending is adding up — track small unplanned purchases."
        )

    if e_pct > 70:
        recommendations.append(
            "Essentials are consuming a large portion of your income. Review subscriptions or recurring bills."
        )

    if status == "Overspending" and not recommendations:
        recommendations.append(
            "Your overall spending is very close to your income. Review recent purchases and reduce non-essential spending."
        )

    if status == "Saving":
        recommendations = [
            "You're spending significantly less than your income. Consider building savings or investing part of it."
        ]

    if not recommendations:
        recommendations.append(
            "Your spending distribution looks healthy. Keep maintaining this balance."
        )

    # -------------------------
    # Dynamic behavior tips
    # -------------------------
    tips = []

    sraw = spender_type_raw.lower()

    if "impulsive" in sraw:
        tips.append("Use the 24-hour rule before making non-essential purchases.")

    elif "saver" in sraw:
        tips.append("You are saving well — consider investing a portion for long-term growth.")

    else:
        tips.append("Track weekly expenses to maintain a balanced budget.")

    # category based hints
    if a_pct > 30:
        tips.append("Plan academic purchases in advance.")

    if l_pct > 20:
        tips.append("Set a weekly limit for leisure activities.")

    if o_pct > 15:
        tips.append("Track miscellaneous expenses carefully.")

    behavior_tip = " ".join(tips)

    return jsonify({
        "essentials_pct": round(e_pct, 2),
        "academic_pct": round(a_pct, 2),
        "leisure_pct": round(l_pct, 2),
        "other_pct": round(o_pct, 2),
        "total_spend_pct": round(total_spend_pct, 2),
        "status": status,
        "recommendations": recommendations,
        "behavior_tip": behavior_tip
    })

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)