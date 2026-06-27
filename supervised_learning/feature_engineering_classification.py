import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Step 1: Load data
df = pd.read_csv(filepath_or_buffer=r"bank-additional-full.csv", sep=';')

# Step 2: Drop leaky feature
df = df.drop(columns=['duration'])

# Step 3: Sentinel handling for pdays
df['was_contacted_before'] = (df['pdays'] != 999).astype(int)
df['pdays_cleaned'] = df['pdays'].replace(999, np.nan)
df['pdays_cleaned'] = df['pdays_cleaned'].fillna(df['pdays_cleaned'].median())
df = df.drop(columns=['pdays'])

# Step 4: Cyclical encoding for month
month_to_num = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
df['month'] = df['month'].str.lower().apply(month_to_num.get)
df['month_sine'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cosine'] = np.cos(2 * np.pi * df['month'] / 12)
job_freq = df['job'].value_counts()
df['job_freq'] = df['job'].map(job_freq)
df = df.drop(columns=['month', 'job'])

# Step 5: Train/test split
X = df.drop(columns=['y'])
Y = (df['y'] == 'yes').astype(np.int32)
print(f"Shape of X: {X.shape}")
print(f"Shape of Y: {Y.shape}")
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, stratify=Y, random_state=42
)

# Step 6: Build ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), make_column_selector(dtype_include=np.number)),
        ('ohe',
            OneHotEncoder(handle_unknown='ignore', drop=None, sparse_output=False),
            make_column_selector(dtype_include="object")
        )
    ],
    remainder='drop',
    verbose_feature_names_out=False
)

# Step 7: Build Pipeline
pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(
            class_weight='balanced', random_state=42
        ))
    ]
)

# Step 8: Fit pipeline
pipeline.fit(X_train, Y_train)

# Step 9: Serialize and reload
joblib.dump(pipeline, 'feature_engineering_classification.joblib', compress=3)
pipe_loaded = joblib.load('feature_engineering_classification.joblib')

# Step 10: Verify round-trip
f1_original = f1_score(Y_test, pipeline.predict(X_test), pos_label=1)
f1_loaded   = f1_score(Y_test, pipe_loaded.predict(X_test), pos_label=1)
assert f1_original == f1_loaded
print(f"Original F1: {f1_original:.4f}")
print(f"Loaded F1:   {f1_loaded:.4f}")
