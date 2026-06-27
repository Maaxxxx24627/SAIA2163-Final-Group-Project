# visualizations/model_comparison.py
import streamlit as st
import pandas as pd
import plotly.express as px

def show_model_comparison():
    """Display comprehensive model comparison chart"""
    st.subheader("🏆 Comprehensive Model Comparison")
    st.markdown("Comparing all 5 trained models across Accuracy, Kappa, and class-specific F1-Scores.")

    # Your data
    model_data = {
        'Model': [
            'Naive Bayes + TF-IDF', 'Naive Bayes + BoW', 'SVM + TF-IDF', 'SVM + BoW', 'Logistic Regression + TF-IDF',
            'Naive Bayes + TF-IDF', 'Naive Bayes + BoW', 'SVM + TF-IDF', 'SVM + BoW', 'Logistic Regression + TF-IDF',
            'Naive Bayes + TF-IDF', 'Naive Bayes + BoW', 'SVM + TF-IDF', 'SVM + BoW', 'Logistic Regression + TF-IDF',
            'Naive Bayes + TF-IDF', 'Naive Bayes + BoW', 'SVM + TF-IDF', 'SVM + BoW', 'Logistic Regression + TF-IDF',
            'Naive Bayes + TF-IDF', 'Naive Bayes + BoW', 'SVM + TF-IDF', 'SVM + BoW', 'Logistic Regression + TF-IDF'
        ],
        'Metric': (['Accuracy'] * 5) + (['Kappa'] * 5) + (['Negative F1'] * 5) + (['Neutral F1'] * 5) + (['Positive F1'] * 5),
        'Score': [
            0.642, 0.612, 0.677, 0.684, 0.669,
            0.373, 0.322, 0.357, 0.366, 0.365,
            0.56, 0.54, 0.55, 0.56, 0.57,
            0.74, 0.70, 0.77, 0.77, 0.75,
            0.42, 0.38, 0.38, 0.38, 0.38
        ]
    }

    df_metrics = pd.DataFrame(model_data)
    model_order = ['SVM + BoW', 'SVM + TF-IDF', 'Logistic Regression + TF-IDF', 'Naive Bayes + TF-IDF', 'Naive Bayes + BoW']

    fig_model_comp = px.bar(
        df_metrics, 
        x='Model', 
        y='Score', 
        color='Metric', 
        barmode='group',
        title='Performance of All Trained Models',
        text='Score',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        category_orders={"Model": model_order}
    )

    fig_model_comp.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig_model_comp.update_layout(
        xaxis_title='Machine Learning Model',
        yaxis_title='Score (0.0 to 1.0)',
        yaxis_range=[0, 0.95],
        legend_title_text='Evaluation Metric',
        hovermode='x unified',
        template='plotly_white',
        height=600
    )

    st.plotly_chart(fig_model_comp, use_container_width=True)

    st.info("""
    💡 **Key Insights from Model Comparison:**
    1. **SVM + BoW** is the overall champion, achieving the highest **Accuracy (68.4%)** and tying for the best **Neutral F1 (0.77)**.
    2. **Logistic Regression + TF-IDF** is the most precise at catching negative sentiment, leading all models with a **Negative F1 of 0.57**.
    3. **The Positive Class Bottleneck:** Across all 5 models, the **Positive F1 score is consistently the lowest (ranging from 0.38 to 0.42)**. This mathematically confirms that the heavy class imbalance (only 37 positive samples in the test set) makes it extremely difficult for any algorithm to reliably predict positive reviews, regardless of the feature extraction method used.
    """)