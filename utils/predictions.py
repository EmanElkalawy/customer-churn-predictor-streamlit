import shap
import pandas as pd
import plotly.graph_objects as go

def get_shap_explanation(model, input_df, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    return explainer, shap_values

def create_gauge(probability):
    prob_percent = probability * 100
    color = "#FF4B4B" if probability > 0.5 else "#00C853"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_percent,
        title={'text': "Churn Probability", 'font': {'size': 28}},
        number={'font': {'size': 72, 'color': color}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40], 'color': "#00C853"},
                {'range': [40, 70], 'color': "#FFB300"},
                {'range': [70, 100], 'color': "#FF4B4B"}
            ]
        }
    ))
    return fig