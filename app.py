import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

@st.cache(allow_output_mutation=True)
def load_data():
    # Load and combine datasets, add debug prints as needed
    combined_df = pd.concat([pd.read_csv(f'Final_Dataset_{year}.csv') for year in range(2020, 2025)]).reset_index(drop=True)
    combined_df['Age'] = pd.to_numeric(combined_df['Age'], errors='coerce')
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = ['1-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
    combined_df['Age_Group'] = pd.cut(combined_df['Age'], bins=bins, labels=labels, right=False)
    
    # Create BMI column
    combined_df['Height_m'] = combined_df['Height'] * 0.3048
    combined_df['BMI'] = combined_df['Weight'] / (combined_df['Height_m'] ** 2)
    
    # Create High_Frequency_Visit column
    combined_df['High_Frequency_Visit'] = (combined_df['Number of Visits'] > 3).astype(int)
    
    return combined_df

df_final = load_data()

st.sidebar.title("Analysis Sections")
section = st.sidebar.selectbox("Choose a section:", ("Introduction", "EDA", "Machine Learning Results"))

if section == "Introduction":
    st.title("Healthcare Data Analysis Dashboard")
    st.write("""
    This dashboard allows you to explore healthcare data ranging from 2020 to 2024.
    - **EDA**: Explore various statistical visualizations.
    - **Machine Learning Results**: Evaluate the performance of different models.
    """)

elif section == "EDA":
    st.title("Exploratory Data Analysis (EDA)")

    # Missing Values Analysis
    st.subheader("Missing Values Analysis")
    missing_values = df_final.isnull().mean() * 100
    fig, ax = plt.subplots(figsize=(12, 8))
    missing_values.sort_values(ascending=False).plot(kind='bar', ax=ax)
    ax.set_title('Percentage of Missing Values by Column')
    ax.set_xlabel('Columns')
    ax.set_ylabel('Percentage of Missing Values')
    st.pyplot(fig)

    # Gender Analysis
    st.subheader("Gender Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x='Gender', data=df_final, palette='coolwarm', ax=ax)
    ax.set_title('Gender Distribution')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    # Vaccination Status Analysis
    st.subheader("Vaccination Status")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x='Vaccination', data=df_final, palette='coolwarm', ax=ax)
    ax.set_title('Vaccination Status')
    ax.set_xlabel('Vaccination')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    # COVID Period Distribution
    st.subheader("COVID Period Distribution")
    df_final['date'] = pd.to_datetime(df_final['date'], errors='coerce')
    df_final['Year'] = df_final['date'].dt.year
    df_final['COVID_Period'] = df_final['Year'].apply(lambda x: 'Pre-COVID' if x in [2020, 2021] else 'Post-COVID' if x in [2022, 2023, 2024] else 'Unknown')
    filtered_df = df_final[df_final['COVID_Period'] != 'Unknown']
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x='COVID_Period', data=filtered_df, palette='coolwarm', ax=ax)
    ax.set_title('Distribution of Pre-COVID and Post-COVID Records')
    ax.set_xlabel('COVID Period')
    ax.set_ylabel('Count')
    st.pyplot(fig)

elif section == "Machine Learning Results":
    st.title("Machine Learning Model Evaluation")
    
    # Prepare data for ML
    features = df_final[['Age', 'BMI', 'Gender', 'Vaccination', 'COVID_Period', 'High_Frequency_Visit']]
    features_final = pd.get_dummies(features, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(features_final.drop('High_Frequency_Visit', axis=1), features_final['High_Frequency_Visit'], test_size=0.2, random_state=42)

    # Train and evaluate models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='linear', probability=True),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        results[name] = {'Accuracy': accuracy, 'ROC AUC': roc_auc}

    # Display Results
    for model_name, result in results.items():
        st.subheader(f"{model_name} Results")
        st.write(f"Accuracy: {result['Accuracy']:.2f}, ROC AUC: {result['ROC AUC']:.2f}")

# Make sure to run this with `streamlit run dashboard.py` from your terminal
