# interactive-plotly-graph
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from boruta import BorutaPy
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Load the data
data = pd.read_csv('/Users/presenter/Desktop/Project/project_data.csv')

# Remove columns with >25% missing values
missing_data = data.isna().sum()
data_clean = data.drop(columns=missing_data[missing_data > len(data) / 4].index)

# Drop irrelevant columns
data_clean = data_clean.drop(columns=['RT', 'SERIALNO'], errors='ignore')

# Convert categorical columns
drop_na_cols = set(data_clean.columns) & set([
    "DIVISION", "SPORDER", "REGION", "STATE", "CIT", "COW", "GCL", "HIMRKS", "HINS1", 
    "HINS2", "HINS3", "HINS4", "HINS5", "HINS6", "HINS7", "LANX", "MAR", "MIG", "MIL", 
    "NWAB", "NWAV", "NWLA", "NWLK", "NWRE", "OIP", "SCH", "SCHL", "SEX", "WRK", "ANC", 
    "ANC1P", "ANC2P", "ESR", "HICOV", "HISP", "INDP", "MSP", "NATIVITY", "OC", "OCCP", 
    "POBP", "PRIVCOV", "PUBCOV", "PUMA", "QTRBIR", "RAC1P", "RAC2P", "RAC3P", "RACAIAN", 
    "RACASN", "RACBLK", "RACNH", "RACNUM", "RACPI", "RACSOR", "RACWHT", "RC", "WAOB", "Class"])
for col in drop_na_cols:
    data_clean[col] = data_clean[col].astype('category')

# Remove near-zero variance features
numeric_cols = data_clean.select_dtypes(include=[np.number])
vt = VarianceThreshold(threshold=0.01)
vt.fit(numeric_cols)
numeric_cols_selected = numeric_cols.columns[vt.get_support()]
data_clean = pd.concat([numeric_cols[numeric_cols_selected], data_clean.select_dtypes(include='category')], axis=1)

# Drop highly correlated numeric variables
corr_matrix = data_clean.select_dtypes(include=[np.number]).corr()
high_corr_var = [(corr_matrix.columns[x], corr_matrix.columns[y])
                 for x, y in zip(*np.where(corr_matrix > 0.8)) if x != y and x < y]
drop_columns_corr = [y for x, y in high_corr_var]
data_clean = data_clean.drop(columns=drop_columns_corr, errors='ignore')

# Scale select columns and remove outliers
z_cols = [c for c in ['PWGTP', 'PERNP', 'PINCP', 'POVPIP'] if c in data_clean.columns]
data_clean[z_cols] = StandardScaler().fit_transform(data_clean[z_cols])
for col in z_cols:
    data_clean = data_clean[np.abs(data_clean[col]) <= 3]

# Fill missing data and save cleaned file
data_clean.ffill(inplace=True)
data_clean.to_csv('/Users/presenter/Desktop/Project/preprocessed_data.csv', index=False)

# Split data
X = data_clean.drop(columns='Class')
y = data_clean['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
X_train.to_csv('/Users/presenter/Desktop/Project/initial_train.csv', index=False)
X_test.to_csv('/Users/presenter/Desktop/Project/initial_test.csv', index=False)

# Balance training data
ros = RandomOverSampler(random_state=123)
X_train_over, y_train_over = ros.fit_resample(X_train, y_train)
rus = RandomUnderSampler(random_state=123)
X_train_under, y_train_under = rus.fit_resample(X_train, y_train)


# Feature Selection
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
selector = RFE(LogisticRegression(max_iter=1000), n_features_to_select=10).fit(X_train_scaled, y_train)
selected_rfe = X_train.columns[selector.support_]

rf = RandomForestClassifier(random_state=123)
boruta = BorutaPy(rf, n_estimators='auto', random_state=123)
boruta.fit(X_train.values, y_train.astype(str).values)
selected_boruta = X_train.columns[boruta.support_]

info_gain = pd.Series(mutual_info_classif(X_train, y_train), index=X_train.columns)
selected_info_gain = info_gain[info_gain > 0].index
###################
# Create individual models with class_weight
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
lr_model = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=1000, random_state=42)
nb_model = GaussianNB()

# Create ensemble
ensemble = VotingClassifier(
    estimators=[('rf', rf_model), ('lr', lr_model), ('nb', nb_model)],
    voting='soft'  # use probability outputs
)

# Fit on training data
ensemble.fit(X_train_scaled, y_train)
################
# Model evaluation loop
models = {
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(random_state=123),
    'Logistic Regression': LogisticRegressionCV(cv=5, max_iter=5000, random_state=123),
    'Gradient Boosting': GradientBoostingClassifier(random_state=123),
    'SVM': SVC(probability=True, random_state=123)
}
sampling_methods = {'Oversampling': ros, 'Undersampling': rus}
feature_sets = {'Info Gain': selected_info_gain, 'Boruta': selected_boruta, 'RFE': selected_rfe}

results = []
for sampling_name, sampler in sampling_methods.items():
    for feature_name, selected_features in feature_sets.items():
        X_train_fs = X_train[selected_features]
        X_test_fs = X_test[selected_features]
        X_resampled, y_resampled = sampler.fit_resample(X_train_fs, y_train)

        for model_name, model in models.items():
            pipe = Pipeline([('scaler', StandardScaler()), ('classifier', model)])
            pipe.fit(X_resampled, y_resampled)
            y_pred = pipe.predict(X_test_fs)
            y_prob = pipe.predict_proba(X_test_fs)[:, 1] if hasattr(model, "predict_proba") else None
            cm = confusion_matrix(y_test, y_pred)
            auc = roc_auc_score(y_test.map({'No': 0, 'Yes': 1}), y_prob) if y_prob is not None else np.nan

            results.append({
                'Sampling': sampling_name,
                'Feature Selection': feature_name,
                'Model': model_name,
                'Confusion Matrix': cm,
                'ROC AUC': round(auc, 4) if not np.isnan(auc) else auc
            })

results_df = pd.DataFrame(results)
print(results_df[['Sampling', 'Feature Selection', 'Model', 'ROC AUC']])

# TPR and TNR calculation
metrics_results = []
for res in results:
    cm = res['Confusion Matrix']
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics_results.append({
            'Sampling': res['Sampling'],
            'Feature Selection': res['Feature Selection'],
            'Model': res['Model'],
            'ROC AUC': res['ROC AUC'],
            'TPR_Yes': round(tpr, 4),
            'TNR_No': round(tnr, 4),
            'Confusion Matrix': cm
        })

metrics_df = pd.DataFrame(metrics_results)
print(metrics_df[['Sampling', 'Feature Selection', 'Model', 'ROC AUC', 'TPR_Yes', 'TNR_No']])

# Best model
best_model = metrics_df.sort_values(by='ROC AUC', ascending=False).iloc[0]
print("\n--- Best Model ---")
print(best_model[['Sampling', 'Feature Selection', 'Model', 'ROC AUC', 'TPR_Yes', 'TNR_No']])

import pandas as pd

# Define your model results
data = {
    "Sampling": [
        "Oversampling"]*18 + ["Undersampling"]*18,
    "Feature Selection": (
        ["Info Gain"]*6 + ["Boruta"]*6 + ["RFE"]*6 +
        ["Info Gain"]*6 + ["Boruta"]*6 + ["RFE"]*6
    ),
    "Model": [
        "KNN", "Naive Bayes", "Random Forest", "Logistic Regression", "Gradient Boosting", "SVM",
        "KNN", "Naive Bayes", "Random Forest", "Logistic Regression", "Gradient Boosting", "SVM",
        "KNN", "Naive Bayes", "Random Forest", "Logistic Regression", "Gradient Boosting", "SVM",
        "KNN", "Naive Bayes", "Random Forest", "Logistic Regression", "Gradient Boosting", "SVM",
        "KNN", "Naive Bayes", "Random Forest", "Logistic Regression", "Gradient Boosting", "SVM",
        "KNN", "Naive Bayes", "Random Forest", "Logistic Regression", "Gradient Boosting", "SVM"
    ],
    "ROC AUC": [
        0.6845, 0.7963, 0.8384, 0.8081, 0.8287, 0.7828,
        0.6175, 0.7053, 0.5902, 0.7311, 0.6105, 0.6932,
        0.6634, 0.7731, 0.7407, 0.8083, 0.7874, 0.7529,
        0.7506, 0.7913, 0.8478, 0.8255, 0.8256, 0.8246,
        0.6272, 0.7103, 0.6289, 0.7350, 0.6139, 0.6749,
        0.7393, 0.7779, 0.7743, 0.8119, 0.7819, 0.7998
    ],
    "TPR_Yes": [
        0.3750, 0.9464, 0.1964, 0.7321, 0.6607, 0.4821,
        0.2679, 0.5714, 0.1429, 0.6250, 0.4107, 0.5714,
        0.2857, 0.9107, 0.4821, 0.7857, 0.7500, 0.7321,
        0.6964, 0.8393, 0.6964, 0.7500, 0.7143, 0.7679,
        0.5357, 0.5714, 0.5714, 0.6250, 0.5179, 0.6071,
        0.6250, 0.9107, 0.7143, 0.7679, 0.6786, 0.7321
    ],
    "TNR_No": [
        0.8925, 0.2272, 0.9918, 0.7810, 0.8517, 0.8939,
        0.8789, 0.7211, 0.8680, 0.7129, 0.7986, 0.7469,
        0.9279, 0.1116, 0.8177, 0.7388, 0.7687, 0.7388,
        0.7156, 0.5551, 0.7551, 0.7537, 0.7769, 0.7293,
        0.6354, 0.7252, 0.6354, 0.7156, 0.7088, 0.7211,
        0.7810, 0.1143, 0.7415, 0.7415, 0.7701, 0.7673
    ]
}

df = pd.DataFrame(data)

# Define highlighting condition for best models
def highlight_best_models(row):
    if row["TPR_Yes"] > 0.81 and row["TNR_No"] > 0.79:
        return ["background-color: #c1f0c1"] * len(row)
    return [""] * len(row)

# Apply style and export to HTML
styled = df.style.apply(highlight_best_models, axis=1)
styled.set_table_attributes("class='table'").to_html("model_results_all.html")

print("✅ HTML file 'model_results_all.html' has been created.")

# Voting Ensemble on Info Gain + Oversampled
X = X_train[selected_info_gain]
y_bin = (y_train == 'Yes').astype(int)
ensemble = Pipeline([
    ('scaler', StandardScaler()),
    ('ensemble', VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000)),
            ('rf', RandomForestClassifier()),
            ('nb', GaussianNB()),
            ('gb', GradientBoostingClassifier())
        ],
        voting='soft',
        n_jobs=-1
    ))
])
ensemble.fit(X, y_bin)
y_probs = ensemble.predict_proba(X)[:, 1]


# Recalculate predictions at threshold = 0.33
threshold = 0.45
y_pred_thresh = (y_probs >= threshold).astype(int)

# Confusion Matrix and Metrics
tn, fp, fn, tp = confusion_matrix(y_bin, y_pred_thresh).ravel()

tpr = tp / (tp + fn)  # True Positive Rate
tnr = tn / (tn + fp)  # True Negative Rate

print(f"\n--- Evaluation at Threshold = {threshold} ---")
print(f"TPR (Sensitivity): {tpr:.4f}")
print(f"TNR (Specificity): {tnr:.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_bin, y_pred_thresh)}")

# Optional: Show precision, recall, f1-score
print("\nClassification Report:")
print(classification_report(y_bin, y_pred_thresh, target_names=["No", "Yes"]))

# Prepare test data using same features
X_test_selected = X_test[selected_info_gain]
y_test_bin = (y_test == 'Yes').astype(int)

# Use pipeline to predict probabilities (scaling handled inside)
y_test_probs = ensemble.predict_proba(X_test_selected)[:, 1]

# Apply threshold
threshold = 0.4
y_test_pred_thresh = (y_test_probs >= threshold).astype(int)

# Evaluate
tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test_bin, y_test_pred_thresh).ravel()

cm = confusion_matrix(y_test_bin, y_test_pred_thresh)
tpr = cm[1, 1] / (cm[1, 1] + cm[1, 0])
tnr = cm[0, 0] / (cm[0, 0] + cm[0, 1])

print(f"\n--- Ensemble Evaluation on Test Set at Threshold = {threshold} ---")
print(f"TPR (Sensitivity): {tpr:.4f}")
print(f"TNR (Specificity): {tnr:.4f}")
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test_bin, y_test_pred_thresh, target_names=['No', 'Yes']))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---- 1. Data ----
data = {
    'Sampling': ['Oversampling'] * 6 + ['Undersampling'] * 6,
    'Feature Selection': ['Info Gain'] * 12,
    'Model': ['KNN', 'Naive Bayes', 'Random Forest', 'Logistic Regression', 'Gradient Boosting', 'SVM'] * 2,
    'TPR_Yes': [0.3750, 0.9464, 0.1964, 0.7321, 0.6607, 0.4821,
                0.6964, 0.8393, 0.6964, 0.7500, 0.7143, 0.7679],
    'TNR_No': [0.8925, 0.2272, 0.9918, 0.7810, 0.8517, 0.8939,
               0.7156, 0.5551, 0.7551, 0.7537, 0.7769, 0.7293]
}
df = pd.DataFrame(data)

after_model = pd.DataFrame([{
    'Sampling': 'Custom Threshold',
    'Feature Selection': 'Final Model',
    'Model': 'Best Tuned Model',
    'TPR_Yes': 0.8205,
    'TNR_No': 0.9922
}])

df_all = pd.concat([df, after_model], ignore_index=True)

# ---- 2. Streamlit UI ----
st.title("Model Performance: TPR vs TNR Analysis")

st.markdown("""
Compare **True Positive Rate (TPR)** and **True Negative Rate (TNR)** across multiple classification models.
Hover over points in the scatterplot for full model details.
""")

# ---- 3. Scatter Plot ----
st.subheader("📊 Interactive Scatterplot: TPR vs TNR")

fig1 = px.scatter(
    df_all,
    x='TPR_Yes',
    y='TNR_No',
    color='Model',
    hover_data=['Sampling', 'Feature Selection'],
    symbol=df_all['Model'].eq('Best Tuned Model'),
    size=df_all['Model'].eq('Best Tuned Model').map({True: 15, False: 8}),
    title="TPR vs TNR for All Models (Interactive)"
)
fig1.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
st.plotly_chart(fig1, use_container_width=True)

# ---- 4. Bar Chart: Before vs After ----
st.subheader("📊 Bar Chart: Before vs After Tuning")

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=['TPR_Yes', 'TNR_No'],
    y=[0.6964, 0.7551],
    name='Best Previous Model',
    marker_color='lightblue'
))
fig2.add_trace(go.Bar(
    x=['TPR_Yes', 'TNR_No'],
    y=[0.8205, 0.9922],
    name='After Tuning',
    marker_color='seagreen'
))
fig2.update_layout(
    title='Before vs After Tuning - TPR and TNR',
    yaxis=dict(title='Score'),
    barmode='group'
)
st.plotly_chart(fig2, use_container_width=True)


# Term_Final_New.py

import streamlit as st
import plotly.express as px
import pandas as pd

# Title of the app
st.title("Interactive Plotly Graph")

# Sample dataset
df = px.data.gapminder().query("year == 2007")

# Sidebar options
continent = st.sidebar.selectbox("Select Continent", df['continent'].unique())

# Filter data
filtered_df = df[df['continent'] == continent]

# Plot
fig = px.scatter(
    filtered_df,
    x="gdpPercap",
    y="lifeExp",
    size="pop",
    color="country",
    hover_name="country",
    log_x=True,
    size_max=60,
    title=f"GDP vs Life Expectancy for {continent} (2007)"
)

st.plotly_chart(fig)


