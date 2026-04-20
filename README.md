# 📉 ChurnGuard - Customer Churn Prediction App

**Live Demo**: [👉 Open App](https://customer-churnguard.streamlit.app/)

**AI-Powered Customer Churn Predictor with Business ROI Calculator**  
Built with XGBoost, scikit-learn Pipeline, SHAP & Streamlit.

A clean, modern, and business-focused ML project ideal for freelancing.

## ✨ Key Features
- Interactive Single Customer Prediction with beautiful gauge
- **Business ROI / Savings Calculator** (shows real monetary value)
- SHAP Explainable AI
- Smart Retention Recommendations
- Clean modern UI with sidebar
- Full production-ready scikit-learn Pipeline

## 🛠 Tech Stack
- **Python** • **XGBoost** • **scikit-learn Pipeline** • **SHAP** • **Streamlit** • **Plotly**

## 📊 Model Performance
- **AUC Score**: 0.8204
- **Dataset**: Telco Customer Churn (Kaggle – 7043 records)

## 🚀 Skills Demonstrated
- End-to-End ML Pipeline Development
- Feature Engineering & Preprocessing
- Explainable AI (SHAP)
- Business Value Translation (ROI Calculator)
- Interactive Web Application (Streamlit)
- Production-ready Code
- Business Value Translation (Retention Strategies)

## 🗂 Project Structure

```
customer-churn-predictor/
├── app.py
├── train_model_improved.py
├── utils/
│   ├── pipeline.py
│   └── predictions.py
├── models/
├── data/
├── requirements.txt
├── README.md
└── screenshots/
```

## 🖥 How to Run Locally

1. Clone the repository
2. Create virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate     # Windows
   source venv/bin/activate  # macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run improved training:
   ```bash
   python train_model_improved.py
   ```
5. Run the app:
   ```bash
   streamlit run app.py
   ```

## 📈 Business Impact
Helps telecom and SaaS companies:
- Identify high-risk customers early
- Reduce churn rate by applying targeted retention strategies
- Save significant revenue through proactive campaigns

## 🔮 Future Enhancements
- Bulk prediction with results download
- Model comparison dashboard
- Docker deployment

---
