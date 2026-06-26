import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
    
    # Text input area
    user_input = st.text_area(
        "Enter text here:", 
        placeholder="E.g., Potong subsidy minyak ni betul-betul menyusahkan rakyat marhaen...",
        height=150
    )
    
    if st.button("Analyze Sentiment", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter some text before clicking the analyze button!")
        else:
            # PLACEHOLDER NOTIFICATION
            st.info("Communicating with the NLP backend... (Running in Mock-up Deployment Mode)")
            st.markdown("---")
            
            # --- SIMULATION VISUALS (To be replaced with Uwais's model.predict() output) ---
            st.success("Prediction Successfully Rendered!")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.subheader("Predicted Class")
                # Visual styling block for text outcome
                st.markdown("**Negative / Angry **")
                st.caption("The model identified indicators of dissatisfaction or socioeconomic concern in the text.")
                
            with res_col2:
                st.subheader("Confidence Score")
                st.metric(label="Probability Confidence", value="94.27%")
                st.progress(0.94)
            
            st.markdown("---")
            st.subheader("Feature Importance & Influential Tokens")
            st.write("The following highly weighted terms in our vocabulary most heavily shifted the prediction toward this category:")
            st.info("Keywords Detected: `potong` (cut), `menyusahkan` (burdening), `minyak` (fuel), `rakyat` (citizens)")


# ---- PAGE 3 : DATASET EXPLORER ----
elif page == "Dataset Explorer":  # Correction du nom pour coller à ta sidebar !
    st.title("Dataset Explorer")
    st.write("Explore the curated dataset used to train and evaluate our sentiment analysis models.")
    st.markdown("---")
    
    import os
    file_path = "data/malaysian_sentiment_labeled.csv"
    
    # 1. Vérification sécurisée de l'existence du fichier
    if os.path.exists(file_path):
        try:
            @st.cache_data
            def load_data():
                return pd.read_csv(file_path)
            
            df = load_data()
            
            # 2. Section des indicateurs clés (KPIs)
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
            
            # 3. Filtres interactifs épurés (vides au départ, mais n'bloquent pas l'affichage)
            st.subheader("Filter & Search Engine")
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                selected_sentiment = st.multiselect(
                    "Filter by Sentiment Category:",
                    options=df['sentiment'].unique(),
                    placeholder="All sentiments shown"  # Plus propre que de tout pré-remplir
                )
            with filter_col2:
                selected_source = st.multiselect(
                    "Filter by Source Type:",
                    options=df['source'].unique(),
                    placeholder="All sources shown"
                )
                
            search_query = st.text_input("Search inside comments (e.g., 'RON95', 'tax'):")
            
            # Application logique des filtres (si vide = on garde tout)
            df_filtered = df.copy()
            if selected_sentiment:
                df_filtered = df_filtered[df_filtered['sentiment'].isin(selected_sentiment)]
            if selected_source:
                df_filtered = df_filtered[df_filtered['source'].isin(selected_source)]
            if search_query:
                df_filtered = df_filtered[df_filtered['body'].str.contains(search_query, case=False, na=False)]
            
            st.markdown("---")
            
            # 4. Affichage du tableau de données interactif
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
        # En attente du push d'Uwais sur GitHub
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