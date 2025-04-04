# -*- coding: utf-8 -*-
"""EDA_Data2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12kpAf6E1bLz8GuUf3wltrO1f9CbMZEup
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = "project_data.csv"
df = pd.read_csv(file_path)

# Basic info and first look
print("\nDataset Info:\n")
print(df.info())
print("\nFirst 5 Rows:\n")
print(df.head())
print("\nSummary Statistics:\n")
print(df.describe(include='all'))
print("\nMissing Values:\n")
print(df.isnull().sum())

# Check for duplicates
print("\nDuplicate Rows:\n")
print(df.duplicated().sum())

# Visualize response type distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='response_type', data=df, palette='viridis')
plt.title('Distribution of Response Types')
plt.show()

# Visualize topic category distribution
plt.figure(figsize=(8, 5))
sns.countplot(y='topic_category', data=df, palette='mako')
plt.title('Distribution of Topic Categories')
plt.show()

# Visualize user intent distribution
plt.figure(figsize=(8, 6))
sns.countplot(y='user_intent', data=df, palette='rocket')
plt.title('Distribution of User Intents')
plt.show()

# Cross-tabulation of response type and topic category
plt.figure(figsize=(8, 6))
sns.countplot(x='response_type', hue='topic_category', data=df, palette='coolwarm')
plt.title('Response Type by Topic Category')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Cross-tabulation of response type and user intent
plt.figure(figsize=(10, 8))
sns.countplot(y='user_intent', hue='response_type', data=df, palette='Set2')
plt.title('Response Type by User Intent')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()