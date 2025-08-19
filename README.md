#ANTI-SOCIAL BEHAVIOR DETECTION IN YOUTUBE COMMENTS USING FASTTEXT

GROUP MEMBERS
Priscilla D. Gborbitey - 22253220
Philip Kwasi Adjartey - 22252449
Nicco-Annan Marilyn - 11410745
Naa Borteley Sebastian-Kumah - 22253153
Bernice Baadawo Abbe – 22253447

EXECUTIVE SUMMARY
This report summarizes a text analytics project focused on detecting toxic comments using machine learning. The project implements a multi-label classification system for identifying six types of toxicity in online comments: toxic, severe_toxic, obscene, threat, insult, and identity_hate. It leverages the Jigsaw Toxic Comment Classification dataset, employing preprocessing techniques, FastText embeddings, and models like Logistic Regression (LR) and Random Forest (RF). The application is built as an interactive Streamlit web app, ensuring stateful navigation, visualizations, and predictions without data leakage. Evaluation metrics indicate strong performance in ROC-AUC, with RF excelling in Hamming Loss and Micro F1, while LR performs better in Macro and Weighted F1.
Key highlights:
Dataset: Train and test CSVs from the Jigsaw challenge, with ~159k training samples showing class imbalance (e.g., ~16k toxic, ~0.5k threats).
Models: LR (OneVsRest with liblinear solver, C=3.0, balanced weights) and RF (300 estimators, random_state=42).
App Features: Interactive dashboard with data previews, preprocessing, modeling, evaluations, and predictions, including EDA visuals like word clouds, n-grams, and label correlations.
Performance: RF shows lower error rates (Hamming Loss: 0.024), while LR has higher AUC (0.963).
This updated report incorporates additional visuals from the app's Dataset Preview section, providing deeper insights into data distributions, and code details on model training. Further files can refine the report.

1. INTRODUCTION
YouTube has become one of the most influential social media platforms, hosting millions of comments daily. However, it faces significant challenges with anti-social behavior such as toxic language, insults, threats, and identity-based hate speech. Manual moderation is impractical due to scale, requiring automated approaches. This project presents a machine learning-based system for detecting anti-social behavior in YouTube comments using FastText embeddings with Logistic Regression and Random Forest classifiers. The system is built as a Streamlit application, offering real-time analysis, model training, and visualization.
2. BACKGROUND
Content moderation is a pressing challenge for online platforms. The dynamic and large-scale nature of user-generated content makes traditional moderation approaches ineffective. Machine learning models can significantly reduce the moderation burden by automatically identifying comments likely to contain toxic content. This project leverages FastText embeddings and multi-label classification to detect six categories of anti-social behavior: toxic, severe_toxic, obscene, threat, insult, and identity_hate.
The system developed herein operationalizes a practical and comprehensive workflow designed to support moderation teams. This workflow spans from initial dataset exploration and rigorous preprocessing to sophisticated modeling, training, evaluation, and prediction, culminating in a deployment-ready inference capability. The objective is to provide a consistent, data-driven screening mechanism that can adapt to the evolving nature of online discourse and the subtle nuances of anti-social behavior. The integration of such a system into existing moderation pipelines promises to streamline operations, improve the overall quality of online interactions, and foster a more secure digital environment for all users.
3. OBJECTIVES
1. Develop an interactive Streamlit application for YouTube comment toxicity detection.
2. Implement a multi-label classification system covering six anti-social categories.
3. Compare Logistic Regression (OvR) and Random Forest models using FastText embeddings.
4. Generate interpretable metrics and actionable insights for effective moderation.
4. DATA OVERVIEW AND EXPLORATION
The Jigsaw Toxic Comment Classification dataset was used, containing 159,571 comments annotated across six toxicity labels. Label imbalance is a significant challenge, with ‘toxic’, ‘insult’, and ‘obscene’ dominating, while ‘threat’ and ‘identity_hate’ are rare. Exploratory data analysis revealed important patterns:
- Length Distribution: Most comments are under 400 characters, with a long-tail distribution.
- Word Cloud: Frequent words include 'one', 'would', 'think', 'dont', and 'people'.
- Label Distribution: Toxic comments dominate, followed by insult and obscene.
- Correlation: Strong positive correlation between insult and obscene (~0.74), and between toxic and insult (~0.65). Threat has very weak correlations.
- N-Gram Analysis: Common bigrams include ‘speed delet’, ‘person attack’, and toxic slurs, highlighting patterns of abusive language.

Dataset
Sources: train.csv (training data with labels) and test.csv (unlabeled test data for predictions).
Labels: Six toxicity columns: toxic, severe_toxic, obscene, threat, insult, identity_hate.
Key Statistics from EDA: 
Training set size: Approximately 159,571 samples (inferred from label counts summing to total with overlaps).
Label Distribution: Highly imbalanced, with "toxic" being the most common (~16,000 instances), followed by "obscene" and "insult" (~8,000 each), "severe_toxic" (~1,500), "identity_hate" (~1,400), and "threat" (~500).
Clean vs. Toxic: Pie chart shows ~90% of comments are clean (non-toxic in any category), with ~10% toxic in at least one label.
Label Co-occurrence: Common overlaps include toxic-obscene (~15,294), toxic-insult (~15,000+), indicating multi-label nature.
Label Correlations: Heatmap reveals positive correlations (e.g., obscene-insult ~0.7, toxic-severe_toxic ~0.3), with minimal negative correlations.
Text Characteristics: 
Comment Length: Histogram shows most comments are short (<500 chars), with a long tail up to ~5,000 chars.
Top Words: Word cloud highlights frequent terms like "one", "think", "dont", "people", "would", "fuck", mixing neutral and potentially toxic language.
Preview Features: The app includes tabs for Overview, Text Analysis (length distribution, word cloud), Label Analysis (distributions, correlations, co-occurrences), and N-gram Explorer (unigrams, bigrams, trigrams) to explore data distributions, correlations, and common phrases.
The app supports caching for efficient loading, and visuals confirm the dataset's challenges like imbalance and toxicity patterns.
5. DATA PREPROCESSING AND PREPARATION
Text preprocessing included lowercasing, removal of URLs, digits, punctuation, and stopwords, as well as stemming using the Porter algorithm. Domain-specific stopwords such as 'wikipedia' and 'edit' were removed. Duplicates and empty comments were dropped. Additional features such as comment length and number of labels were engineered.
6. MODEL DEVELOPMENT AND TRAINING
FastText embeddings were trained with vector size 100, window size 5, min_count 2, and 10 epochs. Two classification models were implemented:
- Logistic Regression (OvR) with class balancing.
- Random Forest with 100 trees.

The dataset was split 80/20 into train and test sets, with fixed random seeds for reproducibility.
7. MODEL PERFORMANCE AND EVALUATION
Evaluation was conducted using F1-scores, ROC-AUC, and Hamming Loss. Key results include:
- Random Forest outperformed in Weighted and Micro F1, and had significantly lower Hamming Loss (0.024).
- Logistic Regression had higher Macro ROC-AUC (0.963 vs 0.938), showing better probability calibration for rare labels.
- Both models performed well on common labels (toxic, insult), but struggled with rare labels like threat.
- Logistic Regression showed relative strength on severe_toxic and identity_hate due to class weighting.
8. IMPLEMENTATION DETAILS
The system was deployed as a Streamlit application, with modules for dataset preview, preprocessing, model training, evaluation, and live comment predictions with confidence scores. Caching was used for performance optimization. Random states and seeds ensured reproducibility.
9. LIMITATIONS AND MITIGATIONS
- Class Imbalance: Mitigated by class weighting, threshold tuning, and recommending human-in-the-loop for rare cases.
- Lack of Interpretability: Suggested use of SHAP/LIME or TF-IDF explainers.
- Static Embeddings: Periodic retraining is necessary to adapt to evolving slang.
- Platform Generalization: Models may need domain adaptation for non-YouTube platforms.
10. CONCLUSIONS
The project successfully built a robust, interpretable system for detecting anti-social behavior in YouTube comments. FastText embeddings provided resilience to noisy text, while Logistic Regression and Random Forest highlighted trade-offs between probability calibration and overall accuracy. The application contributes a scalable tool for content moderation with actionable insights to reduce harm and foster safer online communities.
This project also demonstrates a comprehensive end-to-end text analytics pipeline for toxic comment detection, integrating data handling, ML modeling, and interactive visualization in a Streamlit app. Updated visuals highlight dataset imbalances (example, rare threats) and common toxic phrases, while code reveals tuned model parameters for handling multi-label challenges. Performance metrics show both models are viable, with RF edging out in error minimization and LR in class-balanced scoring.

