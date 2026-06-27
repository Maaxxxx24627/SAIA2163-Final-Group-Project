from email.mime import base

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import re
import os


# 1. PAGE CONFIGURATION & THEME

st.set_page_config(
    page_title="MY Fuel Subsidy Sentiment Analyzer",
    page_icon="🇲🇾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 2. SIDEBAR NAVIGATION

st.sidebar.markdown("# MY SAIA 2163")
st.sidebar.markdown("Final Project Showcase")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Page:",
    ["Home/About", "Text Analyzer", "Dataset Explorer", "Visualizations", "Model Info"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Developed for NLP Application Course")
st.sidebar.caption("June 2026")


# 3. PAGE CONTENT CONTEXTS



# PAGE 1 : HOME / ABOUT

if page == "Home/About":
    st.title("Malaysian Fuel Subsidy Sentiment Analyzer")
    st.markdown("Analyzing Public Response to Socioeconomic Policy Shifts")
    st.markdown("---")
    
    col_intro, col_guide = st.columns([2, 1])
    
    with col_intro:
        st.subheader("Project Overview")
        st.write(
            "Following recent socioeconomic restructuring in Malaysia, the rationalization of "
            "fuel subsidies has sparked significant national discourse. This Natural Language Processing (NLP) "
            "application is designed to automatically capture, clean, and classify public sentiment "
            "and emotional polarities from social media discussions regarding these targeted subsidy cuts."
        )
        st.write(
            "By implementing Machine Learning classification baselines, this tool provides real-time "
            "insights into whether the general public reacts with support, concern, or opposition to "
            "the ongoing financial policies."
        )
        
        st.markdown("Key Features Implemented:")
        st.markdown("- **Real-Time Input Prediction:** Instant sentiment inference from custom text snippets.")
        st.markdown("- **Dataset Explorer Dashboard:** Transparent inspection of the trained dataset distribution.")
        st.markdown("- **Advanced Visualizations:** Insightful analytical graphs including word clouds and confusion matrices.")
        st.markdown("- **Model Evaluation:** Detailed performance metrics comparison between optimized algorithms.")

    with col_guide:
        st.subheader("Application Guide")
        st.info(
            "1. **Analyze Text:** Go to **Text Analyzer** to type or paste any tweet, comment, or headline.\n"
            "2. **Explore Data:** Go to **Dataset Explorer** to view rows and structural stats of our dataset.\n"
            "3. **View Metrics:** Check **Visualizations** and **Model Info** to inspect the algorithmic performance analytics."
        )
        
    st.markdown("---")
    st.subheader("Team Members (Section 4)")
    
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        st.metric(label="Streamlit App Dev", value="MAXENCE")
    with col_t2:
        st.metric(label="NLP Pipeline & ML", value="UWAIS")
    with col_t3:
        st.metric(label="Data Visualizations", value="ZARIF")
    with col_t4:
        st.metric(label="Documentation & Repo", value="ISBULLAH")



# PAGE 2 : TEXT ANALYZER
elif page == "Text Analyzer":
    st.title("Real-Time Sentiment Classifier")
    st.write("Paste a tweet or comment regarding the Malaysian fuel subsidy rationalization below to infer public sentiment.")
    st.markdown("---")
    
    import re
    
    def preprocess_text(text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        text = re.sub(r'[*_]{1,2}([^*_]+)[*_]{1,2}', r'\1', text)
        text = re.sub(r'^>.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'&gt;|&amp;|&lt;|&nbsp;', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'[^\w\s!?]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    try:
        @st.cache_resource
        def load_nlp_models():
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "models", "final_best", "rakyat_speaks_ml_LogReg_TFIDF_FINAL_BEST.pkl")
            vec_path   = os.path.join(base_dir, "models", "final_best", "rakyat_speaks_ml_LogReg_TFIDF_FINAL_BEST_vectorizer.pkl")
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            with open(vec_path, "rb") as f:
                vectorizer = pickle.load(f)
            return vectorizer, model
        
        tfidf_vectorizer, ml_model = load_nlp_models()
        models_loaded = True
    except Exception as e:
        st.error(f"Error loading ML models: {e}")
        models_loaded = False

    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    st.write("**Quick Examples (Click to test):**")
    col_ex1, col_ex2 = st.columns(2)
    with col_ex1:
        if st.button("Example A (Negative Expression)"):
            st.session_state.input_text = "Minyak naik lagi lah aduh pening kepala macam ni. Gomen cuts subsidy burdening rakyat!"
    with col_ex2:
        if st.button("Example B (Neutral Expression)"):
            st.session_state.input_text = "Targeted subsidy RON95 will start soon. Please bring your IC to register at the station."

    user_input = st.text_area(
        "Enter text here:", 
        value=st.session_state.input_text,
        placeholder="Type or click an example above...",
        height=150
    )
    
    if st.button("Analyze Sentiment", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter some text before clicking the analyze button!")
        elif not models_loaded:
            st.error("Cannot analyze. Model files are missing or corrupted.")
        else:
            with st.spinner("Processing text through NLP pipeline..."):
                cleaned_text = preprocess_text(user_input)
                vectorized_text = tfidf_vectorizer.transform([cleaned_text])
                
                raw_prediction = ml_model.predict(vectorized_text)[0]
                prediction_label = str(raw_prediction).upper()
                

                has_proba = hasattr(ml_model, "predict_proba")
                if has_proba:
                    probabilities = ml_model.predict_proba(vectorized_text)[0]
                    class_idx = list(ml_model.classes_).index(raw_prediction)
                    confidence = probabilities[class_idx] * 100
                else:
                    confidence = None

            st.success("Analysis Complete!")
            st.markdown("---")
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.subheader("Predicted Sentiment")
                
                if "POS" in prediction_label:
                    st.markdown(f"### **{prediction_label}**")
                    st.caption("The text reflects support, optimism, or approval toward the policy.")
                elif "NEG" in prediction_label:
                    st.markdown(f"### **{prediction_label}**")
                    st.caption("The text contains indicators of dissatisfaction, anger, or economic concern.")
                else:
                    st.markdown(f"### **{prediction_label}**")
                    st.caption("The text is informative, factual, or lacks explicit emotional polarity.")
                
            with res_col2:
                st.subheader("Model Certainty")
                if confidence is not None:
                    st.metric(label="Probability Confidence", value=f"{confidence:.2f}%")
                    st.progress(confidence / 100)
                else:
                    st.info("Confidence metrics unavailable for this model architecture.")
                    st.metric(label="Prediction Status", value="Verified (Discrete Class)")

            with st.expander("View Pipeline Processing Steps"):
                st.write("**Model Type Loaded:**", type(ml_model).__name__)
                st.write("**Raw Text input:**", user_input)
                st.write("**Cleaned Text (Tokens):**", f"`{cleaned_text}`")


# PAGE 3 : DATASET EXPLORER
elif page == "Dataset Explorer":
    st.title("Dataset Explorer")
    st.write("Explore the curated dataset used to train and evaluate our sentiment analysis models.")
    st.markdown("---")
    
    import os
    file_path = "data/malaysian_sentiment_labeled.csv"
    
    if os.path.exists(file_path):
        try:
            @st.cache_data
            def load_data():
                return pd.read_csv(file_path)
            
            df = load_data()
            
            # 2. KPI Metrics Display
            st.subheader("Dataset Summary")
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            with kpi_col1:
                st.metric(label="Total Data Rows", value=f"{len(df):,}")
            with kpi_col2:
                pos_count = len(df[df['sentiment'].str.lower() == 'positive'])
                st.metric(label="Positive Comments", value=f"{pos_count:,}")
            with kpi_col3:
                neu_count = len(df[df['sentiment'].str.lower() == 'neutral'])
                st.metric(label="Neutral Comments", value=f"{neu_count:,}")
            with kpi_col4:
                neg_count = len(df[df['sentiment'].str.lower() == 'negative'])
                st.metric(label="Negative Comments", value=f"{neg_count:,}")
                
            st.markdown("---")
            
            st.subheader("Filter & Search Engine")
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                selected_sentiment = st.multiselect(
                    "Filter by Sentiment Category:",
                    options=df['sentiment'].unique(),
                    placeholder="All sentiments shown"
                )
            with filter_col2:
                selected_source = st.multiselect(
                    "Filter by Source Type:",
                    options=df['source'].unique(),
                    placeholder="All sources shown"
                )
                
            search_query = st.text_input("Search inside comments (e.g., 'RON95', 'tax'):")
            

            df_filtered = df.copy()
            if selected_sentiment:
                df_filtered = df_filtered[df_filtered['sentiment'].isin(selected_sentiment)]
            if selected_source:
                df_filtered = df_filtered[df_filtered['source'].isin(selected_source)]
            if search_query:
                df_filtered = df_filtered[df_filtered['body'].str.contains(search_query, case=False, na=False)]
            
            st.markdown("---")
            
            st.subheader("Interactive Data Table")
            st.dataframe(
                df_filtered[['body', 'sentiment', 'source', 'topic', 'score']], 
                use_container_width=True,
                height=400
            )
            st.caption(f"Showing {len(df_filtered)} out of {len(df)} entries.")

        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
            
    else:
        st.warning("**Waiting for Uwais' Dataset...**")
        st.info(f"The application is looking for the file at `{file_path}`. Once pushed to GitHub, this page will unlock automatically.")
        
        st.markdown("### Preview of expected structure:")
        mock_data = pd.DataFrame({
            "body": [
                "Targeted subsidy is good for B40, support gomen!", 
                "Minyak naik lagi lah aduh pening kepala macam ni.",
                "Nothing special. subsidi bersasar RON95. Bring your IC."
            ],
            "source": ["comment", "comment", "comment"],
            "topic": ["fuel_subsidy", "fuel_subsidy", "fuel_subsidy"],
            "score": [1, 1, 1],
            "sentiment": ["positive", "negative", "neutral"]
        })
        st.dataframe(mock_data, use_container_width=True)



# PAGE 4 : VISUALIZATIONS

elif page == "Visualizations":
    st.title("Data Insights & Analytical Charts")
    st.write("Exploratory visualizations highlighting data characteristics, vocabulary trends, and baseline evaluation performance.")
    st.markdown("---")

    # ── Chargement des résultats depuis le CSV de Uwais ──────────────────────
    @st.cache_data
    def load_results():
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            return pd.read_csv(os.path.join(base_dir, "results", "model_results.csv"))
        except FileNotFoundError:
            return None

    results_df = load_results()

    # ── 1. CONFUSION MATRICES ─────────────────────────────────────────────────
    st.subheader("1. Confusion Matrices - Final Best Models")
    st.write("True labels vs. predictions for our 3 selected production models (NB + TF-IDF, SVM + BoW, LogReg + TF-IDF).")

    cm_paths = [
        "notebooks/confusion_matrices_final.png",
        "confusion_matrices_final.png",
    ]
    cm_found = False
    for path in cm_paths:
        if os.path.exists(path):
            st.image(path, caption="Confusion Matrix Grid - Final Best Models", use_container_width=True)
            cm_found = True
            break
    if not cm_found:
        st.warning("confusion_matrices_final.png not found - run the notebook to generate it.")

    st.markdown("---")

    # ── 2. CLASS DISTRIBUTION + WORD CLOUD ───────────────────────────────────
    st.subheader("2. Corpus Characteristics")

    vis_col1, vis_col2 = st.columns(2)

    with vis_col1:
        st.markdown("**Sentiment Class Distribution**")
        st.write(
            "The dataset (~7,383 rows) exhibits a heavy natural class skew: "
            "Neutral discussions dominate, while positive sentiment is a minority class (287 rows)."
        )
        dist_paths = ["distribution.png", "notebooks/distribution.png"]
        dist_found = False
        for p in dist_paths:
            if os.path.exists(p):
                st.image(p, caption="Imbalanced Class Distribution (RON95 Dataset)", use_container_width=True)
                dist_found = True
                break
        if not dist_found:
            st.warning("distribution.png not found.")
        st.info("Insight: Class imbalance is the main reason Positive F1 remains low (~0.38) across all models.")

    with vis_col2:
        st.markdown("**Word Cloud - Most Frequent Tokens by Class**")
        st.write(
            "Dominant terms differ significantly across sentiment classes - "
            "revealing distinct vocabulary patterns for each group."
        )
        wc_tab1, wc_tab2, wc_tab3 = st.tabs(["Negative", "Neutral", "Positive"])
        with wc_tab1:
            wc_paths = ["notebooks/wordcloud_negative.png", "wordcloud_negative.png"]
            for p in wc_paths:
                if os.path.exists(p):
                    st.image(p, use_container_width=True)
                    break
        with wc_tab2:
            wc_paths = ["notebooks/wordcloud_neutral.png", "wordcloud_neutral.png"]
            for p in wc_paths:
                if os.path.exists(p):
                    st.image(p, use_container_width=True)
                    break
        with wc_tab3:
            wc_paths = ["notebooks/wordcloud_positive.png", "wordcloud_positive.png"]
            for p in wc_paths:
                if os.path.exists(p):
                    st.image(p, use_container_width=True)
                    break
        st.caption("Regional stopwords filtered to extract root semantics.")
        
    st.markdown("---")

    # ── 3. MODEL COMPARISON ───────────────────────────────────────────────────
    st.subheader("3. Model Performance Comparison")
    st.write("Accuracy and Kappa score across all evaluated models - final best highlighted.")

    comp_paths = ["model_comparison.png", "notebooks/model_comparison.png"]
    comp_found = False
    for p in comp_paths:
        if os.path.exists(p):
            st.image(p, caption="Model Accuracy & Kappa Comparison", use_container_width=True)
            comp_found = True
            break

    if not comp_found:
        if results_df is not None:
            final = results_df[results_df['status'] == 'final_best'].copy()
            fallback_models = final['model'].tolist()
            fallback_acc    = (final['accuracy'] * 100).tolist()
            fallback_kappa  = final['kappa'].tolist()
        else:
            fallback_models = ['NB + TF-IDF', 'SVM + BoW', 'LR + TF-IDF']
            fallback_acc    = [64.2, 65.4, 66.9]
            fallback_kappa  = [0.373, 0.358, 0.365]

        fig, ax1 = plt.subplots(figsize=(10, 5))

        x = np.arange(len(fallback_models))
        width = 0.35

        bars1 = ax1.bar(x - width / 2, fallback_acc, width, label='Accuracy (%)', color='steelblue')
        ax1.set_ylabel('Accuracy (%)', color='steelblue')
        ax1.set_ylim(0, 100)
        ax1.tick_params(axis='y', labelcolor='steelblue')

        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + width / 2, fallback_kappa, width, label="Cohen's Kappa", color='coral')
        ax2.set_ylabel("Cohen's Kappa", color='coral')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='y', labelcolor='coral')

        ax1.set_xticks(x)
        ax1.set_xticklabels(fallback_models, rotation=15)
        ax1.set_title('Model Comparison - Final Best Models')

        lines = [bars1, bars2]
        labels = ['Accuracy (%)', "Cohen's Kappa"]
        ax1.legend(lines, labels, loc='upper left')

        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Generated dynamically from results/model_results.csv - add model_comparison.png to use static version.")

    st.markdown("---")

    # ── 4. TOP 20 MOST COMMON WORDS ───────────────────────────────────────────
    st.subheader("4. Top 20 Most Common Words")
    st.write("Most frequent tokens in the full corpus after stopword removal and preprocessing.")

    top20_paths = ["top20_words.png", "notebooks/top20_words.png"]
    t20_found = False
    for p in top20_paths:
        if os.path.exists(p):
            st.image(p, caption="Top 20 Words by Frequency", use_container_width=True)
            t20_found = True
            break
    if not t20_found:
        st.warning("top20_words.png not found - Zarif needs to generate this from the dataset.")

    st.markdown("---")

    # ── 5. PER-CLASS F1 BREAKDOWN ─────────────────────────────────────────────
    st.subheader("5. Per-Class F1 Score Breakdown")
    st.write("F1 performance per sentiment class - reveals the impact of class imbalance on the minority Positive class.")

    f1_paths = ["f1_breakdown.png", "notebooks/f1_breakdown.png"]
    f1_found = False
    for p in f1_paths:
        if os.path.exists(p):
            st.image(p, caption="Per-Class F1 Score by Model", use_container_width=True)
            f1_found = True
            break

    if not f1_found:
        if results_df is not None:
            final = results_df[results_df['status'] == 'final_best'].copy()
            f1_models  = final['model'].tolist()
            f1_neg     = final['neg_f1'].tolist()
            f1_neu     = final['neu_f1'].tolist()
            f1_pos     = final['pos_f1'].tolist()
        else:
            f1_models = ['NB + TF-IDF', 'SVM + BoW', 'LR + TF-IDF']
            f1_neg    = [0.556, 0.560, 0.571]
            f1_neu    = [0.738, 0.745, 0.754]
            f1_pos    = [0.416, 0.400, 0.381]

        fig2, ax2 = plt.subplots(figsize=(11, 5))
        x = np.arange(len(f1_models))
        width = 0.25
        ax2.bar(x - width, f1_neg, width, label='Negative F1', color='#e74c3c')
        ax2.bar(x,         f1_neu, width, label='Neutral F1',  color='#3498db')
        ax2.bar(x + width, f1_pos, width, label='Positive F1', color='#2ecc71')
        ax2.set_xticks(x)
        ax2.set_xticklabels(f1_models, rotation=15)
        ax2.set_ylabel('F1 Score')
        ax2.set_ylim(0, 1)
        ax2.set_title('Per-Class F1 Score - Final Best Models')
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)
        st.caption("Positive F1 remains low (~0.38–0.42) due to severe underrepresentation (287 positive vs 5,033 neutral rows).")
        st.info("Insight: This imbalance is the core argument for testing transformer models - BERT/RoBERTa handle minority classes more robustly through contextual embeddings.")# PAGE 5 : MODEL INFO


# PAGE 5 : MODEL INFO

elif page == "Model Info":
    st.title("Model Architecture & Evaluation Breakdown")
    st.write("Detailed comparative look into the hyper-parameters and metrics of our trained pipeline models.")
    st.markdown("---")

    @st.cache_data
    def load_results_model_info():
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            return pd.read_csv(os.path.join(base_dir, "results", "model_results.csv"))
        except FileNotFoundError:
            return None

    results_df = load_results_model_info()

    st.subheader("Comparative Testing Performance Grid")
    st.write("Evaluation results gathered over the 402 gold-labeled testing split entries:")

    if results_df is not None:
        display_df = results_df.copy()
        display_df['accuracy'] = (display_df['accuracy'] * 100).round(1).astype(str) + '%'
        display_df['kappa']    = display_df['kappa'].round(3).astype(str)
        display_df['neg_f1']   = display_df['neg_f1'].round(3).astype(str)
        display_df['neu_f1']   = display_df['neu_f1'].round(3).astype(str)
        display_df['pos_f1']   = display_df['pos_f1'].round(3).astype(str)
        display_df = display_df.rename(columns={
            'model':    'Model',
            'status':   'Status',
            'accuracy': 'Accuracy',
            'kappa':    "Cohen's Kappa",
            'neg_f1':   'Negative F1',
            'neu_f1':   'Neutral F1',
            'pos_f1':   'Positive F1'
        })
        st.dataframe(display_df, use_container_width=True)

        # Best model dynamique
        best_row = results_df.loc[results_df['kappa'].idxmax()]
        best_name = best_row['model']
        best_acc  = best_row['accuracy'] * 100
        best_kappa = best_row['kappa']
    else:
        st.warning("model_results.csv not found — showing static fallback data.")
        fallback = pd.DataFrame({
            'Model': ['Naive Bayes + TF-IDF', 'SVM + BoW', 'Logistic Regression + TF-IDF'],
            'Status': ['final_best', 'final_best', 'final_best'],
            'Accuracy': ['64.2%', '68.4%', '66.9%'],
            "Cohen's Kappa": ['0.373', '0.366', '0.365'],
            'Negative F1': ['0.556', '0.556', '0.571'],
            'Neutral F1': ['0.738', '0.772', '0.754'],
            'Positive F1': ['0.416', '0.384', '0.381'],
        })
        st.dataframe(fallback, use_container_width=True)
        best_name  = "SVM + BoW"
        best_acc   = 68.4
        best_kappa = 0.366

    st.markdown("---")

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown("**Text Preprocessing Architecture:**")
        st.write("- **Character Normalization:** Lowercasing, removal of URLs, usernames, markdown symbols, and non-ASCII characters.")
        st.write("- **Token Cleaning:** Regex filtering to strip punctuation while preserving `!` and `?` for sentiment extraction.")
        st.write("- **Length Threshold:** Automatic deletion of text fragments shorter than 3 tokens to minimize dataset noise.")
        st.write("- **Feature Scope:** Extraction bounds capped at 30,000 top n-grams (1,2) with minimum document frequency (`min_df=2`).")

    with col_p2:
        st.markdown("**Best Performer Evaluation:**")
        st.success(f"Best Overall Model: **{best_name}** ({best_acc:.1f}% Acc, Kappa {best_kappa:.3f})")
        st.write(
            "All models face a linguistic hurdle with the **Positive Class F1 (~0.38–0.42)** "
            "due to extreme dataset imbalance — only 287 positive training rows vs 5,033 neutral."
        )
        st.info(
            "Training was optimized using balanced sample weights (`class_weight='balanced'`) "
            "to prevent algorithms from ignoring minority positive expressions. "
            "Transformer models (MalayBERT, XLM-R) were subsequently tested to address this imbalance."
        )

    st.markdown("---")
    st.subheader("Model Deployment Artifacts")
    st.write("Production models serialized from Uwais' pipeline — loaded at runtime by the Text Analyzer:")

    dep_col1, dep_col2 = st.columns(2)
    with dep_col1:
        st.code("models/final_best/rakyat_speaks_ml_LogReg_TFIDF_FINAL_BEST.pkl", language="text")
        st.caption("Logistic Regression weights — primary inference model for live predictions.")
    with dep_col2:
        st.code("models/final_best/rakyat_speaks_ml_LogReg_TFIDF_FINAL_BEST_vectorizer.pkl", language="text")
        st.caption("TF-IDF vocabulary mapping — converts raw text to feature vectors.")