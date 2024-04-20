from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
import numpy as np  #numerical calculations
import pandas as pd #data frames
import matplotlib.pyplot as plt  #visualization
import seaborn as sns #visualization

#loading datasets from 2020 to 2024
df1 = pd.read_csv('Final_Dataset_2020.csv')
df2 = pd.read_csv('Final_Dataset_2021.csv')
df3 = pd.read_csv('Final_Dataset_2022.csv')
df4 = pd.read_csv('Final_Dataset_2023.csv')
df5 = pd.read_csv('Final_Dataset_2024.csv')

# Merging five datasets
combined_df = pd.concat([df1, df2, df3, df4], axis=0)

# Write the combined DataFrame to a new CSV file
combined_df.to_csv('data.csv', index=False)

df_final= pd.read_csv('data.csv')

df_final['High_Frequency_Visit'] = (df_final['Number of Visits'] > 3).astype(int)
features = df_final[['Age', 'BMI', 'Gender', 'Vaccination']]# Convert categorical data to dummy variables
features_final = pd.get_dummies(df_final[['Age', 'BMI', 'Gender', 'Vaccination']], drop_first=True)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final, df_final['High_Frequency_Visit'], test_size=0.2, random_state=42)
# Initialize the models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    #"SVM": SVC(kernel='linear', probability=True),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    results[name] = {'Accuracy': accuracy, 'ROC AUC': roc_auc}

# Print results
for model in results:
    print(f"{model}:\n Accuracy: {results[model]['Accuracy']:.2f}, ROC AUC: {results[model]['ROC AUC']:.2f}\n")model_names = ['Logistic Regression', 'Random Forest', 'SVM', 'Gradient Boosting']
accuracies = [0.71, 0.65, 0.71, 0.70]  
roc_aucs = [0.62, 0.53, 0.61, 0.61]  

x = np.arange(len(model_names))  
width = 0.35  

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, accuracies, width, label='Accuracy')
rects2 = ax.bar(x + width/2, roc_aucs, width, label='ROC AUC')

ax.set_xlabel('Models')
ax.set_title('Comparison of Machine Learning Models')
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
fig.tight_layout()

plt.show()