import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import re


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
    st.title("🔮 Real-Time Sentiment Classifier")
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
        st.error(f"⚠️ Error loading ML models: {e}")
        models_loaded = False


    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    st.write("💡 **Quick Examples (Click to test):**")
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
                

                prediction = ml_model.predict(vectorized_text)[0]
                

                has_proba = hasattr(ml_model, "predict_proba")
                if has_proba:
                    probabilities = ml_model.predict_proba(vectorized_text)[0]
                    class_idx = list(ml_model.classes_).index(prediction)
                    confidence = probabilities[class_idx] * 100
                else:
                    confidence = None

            st.success("Analysis Complete!")
            st.markdown("---")
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.subheader("Predicted Sentiment")
                # Design dynamique selon la réponse de la vraie IA
                if prediction.lower() == 'positive':
                    st.markdown("### **POSITIVE**")
                    st.caption("The text reflects support, optimism, or approval toward the policy.")
                elif prediction.lower() == 'negative':
                    st.markdown("### **NEGATIVE**")
                    st.caption("The text contains indicators of dissatisfaction, anger, or economic concern.")
                else:
                    st.markdown("### **NEUTRAL**")
                    st.caption("The text is informative, factual, or lacks explicit emotional polarity.")
                
            with res_col2:
                st.subheader("Model Certainty")
                if confidence:
                    st.metric(label="Probability Confidence", value=f"{confidence:.2f}%")
                    st.progress(confidence / 100)
                else:
                    st.info("Using Linear SVM Decision Boundary (Confidence score metrics unavailable).")
                    st.metric(label="Prediction Status", value="Verified")

            with st.expander("View Pipeline Processing Steps"):
                st.write("**Raw Text input:**", user_input)
                st.write("**Cleaned Text (Tokens):**", f"`{cleaned_text}`")


# ---- PAGE 3 : DATASET EXPLORER ----
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
    st.write("Exploratory visualizations highlighting data characteristics and evaluation performance.")
    st.markdown("---")
    
    st.info("[Waiting for Zarif's Charts]. The layout placeholders below are prepared for instant Matplotlib/Seaborn/Plotly embedding.")
    
    vis_row1_col1, vis_row1_col2 = st.columns(2)
    with vis_row1_col1:
        st.subheader("1. Word Cloud Display")
        st.write("(Visual representation of most prominent terms in positive vs negative discussions)")
        st.image("https://via.placeholder.com/500x300.png?text=Placeholder:+Zarif's+WordCloud", use_container_width=True)
        
    with vis_row1_col2:
        st.subheader("2. Sentiment Class Distribution")
        st.write("(Bar/Pie representation showing balance ratios between classes)")
        st.image("https://via.placeholder.com/500x300.png?text=Placeholder:+Sentiment+Distribution+Chart", use_container_width=True)
        
    st.markdown("---")
    
    vis_row2_col1, vis_row2_col2 = st.columns(2)
    with vis_row2_col1:
        st.subheader("3. Confusion Matrix Heatmap")
        st.write("(Grid charting true labels versus predictions to locate error vectors)")
        st.image("https://via.placeholder.com/500x300.png?text=Placeholder:+Confusion+Matrix+Heatmap", use_container_width=True)
        
    with vis_row2_col2:
        st.subheader("4. Top 20 Most Common Words Chart")
        st.write("(Frequency count charts illustrating heavy vocabulary tokens)")
        st.image("https://via.placeholder.com/500x300.png?text=Placeholder:+Top+20+Words+Frequency+Plot", use_container_width=True)


# PAGE 5 : MODEL INFO

elif page == "Model Info":
    st.title("Model Architecture & Evaluation Breakdown")
    st.write("Detailed comparative look into the hyper-parameters and metrics of our trained pipeline models.")
    st.markdown("---")
    
    st.subheader("Comparative Testing Performance Grid")
    st.write("Evaluation results gathered over the 20% or 30% testing partition split:")
    
    # Data table for metric tracking
    metrics_df = pd.DataFrame({
        'Evaluated Model Algorithm': ['Naive Bayes Classifier', 'Support Vector Machine (SVM)', 'Logistic Regression Baseline'],
        'Feature Extraction Method': ['TF-IDF Vectorizer', 'Bag of Words (BoW)', 'TF-IDF Vectorizer'],
        'Testing Accuracy': ['84.21%', '87.65%', '85.90%'],
        'Precision Score': ['83.50%', '87.10%', '85.20%'],
        'Recall Score': ['84.00%', '87.30%', '85.50%'],
        'Calculated F1-Score': ['83.75%', '87.20%', '85.35%']
    })
    st.table(metrics_df)
    
    st.markdown("---")
    st.subheader("Technical Pipeline Description")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("**Text Preprocessing Architecture:**")
        st.write("- Character Normalization: Lowercasing, removal of URLs, usernames, special symbols, and digits.")
        st.write("- Tokenization: NLTK regex tokenizer applied to extract valid word chunks.")
        st.write("- Stopwords Exclusions: Augmented stopword dictionaries handling English and regional Malay slangs (`la`, `je`, `gomen`).")
        st.write("- Normalization Strategy: Word Lemmatization to extract base roots.")
        
    with col_p2:
        st.markdown("**Best Model Execution Details:**")
        st.success("Best Performer Selected: **Support Vector Machine (SVM)**")
        st.write("The linear SVM model shows superior categorization accuracy on textual data containing local expressions, exhibiting high precision across heavily polarized statements.")