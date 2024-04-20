import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px

def load_data():
    file_names = ['Final_Dataset_2020.csv', 'Final_Dataset_2021.csv', 
                  'Final_Dataset_2022.csv', 'Final_Dataset_2023.csv', 'Final_Dataset_2024.csv']
    data_frames = []
    for file in file_names:
        df = pd.read_csv(file)
        data_frames.append(df)
    
    combined_df = pd.concat(data_frames, axis=0, ignore_index=True)

    # Convert 'Date' column to datetime if it exists
    if 'Date' in combined_df.columns:
        combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce')
    else:
        st.error("Date column not found in the dataset. Please check the dataset.")

    # Check if 'Height' and 'Weight' are in columns for BMI calculation
    if 'Height' in combined_df.columns and 'Weight' in combined_df.columns:
        combined_df['Height_m'] = combined_df['Height'] * 0.01  # Assuming height is in cm
        combined_df['BMI'] = combined_df['Weight'] / (combined_df['Height_m'] ** 2)
    else:
        st.error("Height and/or Weight columns not found for BMI calculation.")

    return combined_df


def plot_correlation(df):
    correlation_matrix = df[['Age', 'Weight', 'Height', 'BMI']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot(plt)

def plot_bmi_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['BMI'], kde=True)
    st.pyplot(plt)

def plot_gender_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Gender', data=df, palette='coolwarm')
    st.pyplot(plt)

def plot_vaccination_status(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Vaccination', data=df, palette='coolwarm')
    st.pyplot(plt)

def generate_wordcloud(df, column):
    text = " ".join(symptom for symptom in df[column].dropna())
    wordcloud = WordCloud(width=800, height=400).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

def main():
    st.title("Healthcare Data Analysis Dashboard")
    df = load_data()

    analysis_options = ["Correlation Matrix", "BMI Distribution", "Gender Distribution", 
                        "Vaccination Status", "Symptoms Word Cloud"]
    analysis_type = st.sidebar.selectbox("Select Analysis Type", analysis_options)

    if analysis_type == "Correlation Matrix":
        plot_correlation(df)
    elif analysis_type == "BMI Distribution":
        plot_bmi_distribution(df)
    elif analysis_type == "Gender Distribution":
        plot_gender_distribution(df)
    elif analysis_type == "Vaccination Status":
        plot_vaccination_status(df)
    elif analysis_type == "Symptoms Word Cloud":
        generate_wordcloud(df, 'Symptoms')

if __name__ == "__main__":
    main()
