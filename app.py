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
            with open("models/tfidf_vectorizer.pkl", "rb") as f:
                vectorizer = pickle.load(f)
            with open("models/best_model.pkl", "rb") as f:
                model = pickle.load(f)
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



# PAGE 4 : VISUALIZATIONS (VERSION DYNAMIQUE)

elif page == "Visualizations":
    st.title("Data Insights & Analytical Charts")
    st.write("Exploratory visualizations highlighting data characteristics, vocabulary trends, and baseline evaluation performance.")
    st.markdown("---")
    
    import os
    import pickle

    st.subheader("1. Model Performance: Confusion Matrix")
    st.write(
        "The matrix below adapts dynamically to showcase the true labels versus predictions for your "
        "currently loaded model architecture."
    )

    # Détection automatique du modèle chargé
    cm_filename = "confusion_matrices.png"
    cm_caption = "Comprehensive Confusion Matrix Breakdown"
    
    try:
        # On inspecte discrètement le modèle pour savoir qui il est
        with open("models/best_model.pkl", "rb") as f:
            check_model = pickle.load(f)
            model_name = type(check_model).__name__.lower()
            
        # On adapte l'image selon la nature de la structure détectée
        if "logistic" in model_name:
            cm_filename = "cm_logistic_tfidf.png"
            cm_caption = "Dynamic Visualization: Logistic Regression + TF-IDF Confusion Matrix"
        elif "nb" in model_name or "naive" in model_name or "bayes" in model_name:
            cm_filename = "cm_nb_tfidf.png"
            cm_caption = "Dynamic Visualization: Naive Bayes + TF-IDF Confusion Matrix"
    except Exception:
        # En cas d'absence du fichier modèle lors des tests locaux
        pass

    # Logique d'affichage de la matrice
    cm_paths = [cm_filename, f"notebooks/{cm_filename}", "confusion_matrices.png", "notebooks/confusion_matrices.png"]
    cm_found = False

    for path in cm_paths:
        if os.path.exists(path):
            st.image(path, caption=cm_caption, use_container_width=True)
            cm_found = True
            break

    if not cm_found:
        st.warning("Confusion matrix chart not found yet.")
        st.image("https://via.placeholder.com/1000x500.png?text=Waiting+for+Zarif+Confusion+Matrices", use_container_width=True)

    st.markdown("---")
    
    st.subheader("2. Corpus Characteristics & Class Distributions")
    
    vis_col1, vis_col2 = st.columns(2)
    
    with vis_col1:
        st.markdown("**Sentiment Class Distribution**")
        st.write(
            "Our curated dataset (~7,383 rows) exhibits a heavy natural class skew: "
            "Neutral/Informative discussions represent the vast majority, while highly polarized positive feedback "
            "remains a minority class (287 rows)."
        )
        
        dist_paths = ["distribution.png", "notebooks/distribution.png"]
        dist_found = False
        for p in dist_paths:
            if os.path.exists(p):
                st.image(p, caption="Imbalanced Class Distribution (RON95 Dataset)", use_container_width=True)
                dist_found = True
                break
        if not dist_found:
            st.image("https://via.placeholder.com/500x350.png?text=Placeholder:+Distribution+Chart", use_container_width=True)
            
        st.info("Insight for presentation: Class imbalance naturally lowers the global Positive F1-score to ~0.38 across all models.")
        
    with vis_col2:
        st.markdown("**Word Cloud & Vocabulary Weighting**")
        st.write(
            "Prominent lexical tokens detected in complaints heavily center around financial pressure terms "
            "like `potong` (cut), `menyusahkan` (burdening), and `harga` (price). Factual records focus on structural keywords."
        )
        
        wc_paths = ["wordcloud.png", "notebooks/wordcloud.png"]
        wc_found = False
        for p in wc_paths:
            if os.path.exists(p):
                st.image(p, caption="Most Frequent Tokens (Stop-words Filtered)", use_container_width=True)
                wc_found = True
                break
        if not wc_found:
            st.image("https://via.placeholder.com/500x350.png?text=Placeholder:+Word+Cloud+Image", use_container_width=True)
            
        st.caption("Text pre-processing accurately filtered out regional stop-words (`la`, `je`, `gomen`) to extract root semantics.")


# PAGE 5 : MODEL INFO

elif page == "Model Info":
    st.title("Model Architecture & Evaluation Breakdown")
    st.write("Detailed comparative look into the hyper-parameters and metrics of our trained pipeline models.")
    st.markdown("---")
    
    st.subheader("Comparative Testing Performance Grid")
    st.write("Evaluation results gathered over the 402 gold-labeled testing split entries:")
    
    metrics_df = pd.DataFrame({
        'Evaluated Model Algorithm': [
            'Naive Bayes Classifier', 
            'Naive Bayes Classifier', 
            'Logistic Regression Baseline', 
            'Logistic Regression Baseline'
        ],
        'Feature Extraction Method': [
            'TF-IDF Vectorizer', 
            'Bag of Words (BoW)', 
            'TF-IDF Vectorizer', 
            'Bag of Words (BoW)'
        ],
        'Testing Accuracy': ['64.2%', '61.2%', '66.9%', '67.7%'],
        'Cohen\'s Kappa Score': ['0.373', '0.322', '0.365', '0.354'],
        'Negative F1-Score': ['0.556', '0.543', '0.571', '0.546'],
        'Neutral F1-Score': ['0.738', '0.701', '0.754', '0.765'],
        'Positive F1-Score': ['0.416', '0.378', '0.381', '0.386']
    })
    
    st.dataframe(metrics_df, use_container_width=True)
    
    st.markdown("---")
    
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        st.markdown("**Text Preprocessing Architecture:**")
        st.write("- **Character Normalization:** Lowercasing, removal of URLs, usernames, markdown symbols, and non-ASCII characters.")
        st.write("- **Token Cleaning:** Regex filtering to strip punctuation while preserving punctuation indicators like `!` and `?` for sentiment extraction.")
        st.write("- **Length Threshold:** Automatic deletion of text fragments shorter than 3 tokens to minimize dataset noise.")
        st.write("- **Feature Scope:** Extraction bounds capped at a maximum of 30,000 top n-grams (1,2) with a minimum document frequency (`min_df=2`).")
        
    with col_p2:
        st.markdown("**Best Performer Evaluation:**")
        st.success("Best Overall Model: **Logistic Regression + BoW** (67.7% Acc)")
        st.write(
            "While **Logistic Regression** achieves the highest raw accuracy, all models face a massive linguistic hurdle "
            "with the **Positive Class F1-Score (~38%)** due to the extreme dataset imbalance (only 287 positive training rows)."
        )
        st.info(
            "*Soutenance Note:* We optimized training by calculating balanced sample weights (`class_weight='balanced'`) "
            "to prevent the algorithms from completely ignoring minority positive expressions."
        )

    st.markdown("---")
    st.subheader("Model Deployment Artifacts")
    st.write("The current production environment is powered by the static serialization bins generated from Uwais' pipeline:")
    
    dep_col1, dep_col2 = st.columns(2)
    with dep_col1:
        st.code("models/best_model.pkl", language="text")
        st.caption("Serialized Python object holding the trained Logistic Regression weights.")
    with dep_col2:
        st.code("models/tfidf_vectorizer.pkl", language="text")
        st.caption("Vocab mapping array converting live text inputs into predictable numeric coordinates.")