# Importing Libraries 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler

# Loading Dataset
df = pd.read_csv("C:/Users/Acer/Downloads/CreditCard_data.csv")
x = df.drop("Class", axis = 1)
y = df["Class"]

# Balancing Dataset (Oversampling)
ros = RandomOverSampler(random_state = 42)
x_balanced, y_balanced = ros.fit_resample(x, y)
balanced_df = pd.DataFrame(x_balanced, columns = x.columns)
balanced_df["Class"] = y_balanced

# Sampling
def simple_random_sampling(df, frac = 0.6):
    return df.sample(frac = frac, random_state = 42)

def systematic_sampling(df, step = 2):
    return df.iloc[::step]

def stratified_sampling(df, frac = 0.6):
    return df.groupby("Class", group_keys = False).apply(lambda x: x.sample(frac = frac, random_state = 42))

def cluster_sampling(df):
    df = df.copy()
    df["Cluster"] = df.index % 5
    choosen_cluster = np.random.choice(df["Cluster"].unique())
    return df[df["Cluster"] == choosen_cluster].drop("Cluster", axis = 1)

def bootstrap_sampling(df, n_samples):
    return df.sample(n = n_samples, replace = True, random_state = 42)

# Creating Samples
samples = {
    "Simple Random": simple_random_sampling(balanced_df),
    "Systematic": systematic_sampling(balanced_df),
    "Stratified": stratified_sampling(balanced_df),
    "Cluster": cluster_sampling(balanced_df),
    "Bootstrap": bootstrap_sampling(balanced_df, len(balanced_df))
}

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter = 1000),
    "Decision Tree": DecisionTreeClassifier(random_state = 42),
    "Random Forest": RandomForestClassifier(n_estimators = 100, random_state = 42),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC()
}

# Training and Evaluation
results = []
for sample_name, sample_df in samples.items():
    x = sample_df.drop("Class", axis = 1)
    y = sample_df["Class"]
    scaler = StandardScaler()
    x_sclaed = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_sclaed, y, test_size = 0.2, random_state = 42, stratify = y)
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred) * 100
        results.append({
            "Sampling Technique": sample_name,
            "Model": model_name,
            "Accuracy": accuracy
        })

# Accuracy Table
results_df = pd.DataFrame(results)
accuracy_table = (
    results_df
    .pivot(index = "Model", columns = "Sampling Technique", values = "Accuracy")
    .round(2)
)
print("\n Accuracy Table (Sampling x Model):")
print(accuracy_table)

# Best Sampling Technique per Model
best_sampling = accuracy_table.idxmax(axis = 1)
print("\n Best Sampling Technique per Model:")
print(best_sampling)