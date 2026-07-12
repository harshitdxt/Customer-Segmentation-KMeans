# Customer Segmentation using K-Means

## Overview

In this project, I used the K-Means Clustering algorithm to group customers based on their annual income and spending score. Since this is an unsupervised learning problem, there is no target column. The model automatically finds similar customers and creates different customer segments.

## Dataset

- Rows: 200
- Columns: 5

Features:
- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

## Steps Performed

- Loaded and explored the dataset
- Checked missing values
- Created distribution plots
- Visualized Annual Income vs Spending Score
- Used the Elbow Method to find the best value of K
- Trained a K-Means model
- Visualized customer clusters
- Saved the clustered dataset

## Key Learning

- Difference between Supervised and Unsupervised Learning
- K-Means Clustering
- Elbow Method
- Customer Segmentation
- Cluster Visualization

## Tech Stack

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn