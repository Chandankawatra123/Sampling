import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import accuracy_score
# Load the original dataset from CSV
file_path = r'C:\Users\91896\Desktop\SamplinG\Creditcard_data.csv'
df = pd.read_csv(file_path)

# Separate the classes
fraudulent = df[df['Class'] == 1]
non_fraudulent = df[df['Class'] == 0]

# Subsample the majority class
non_fraudulent_sampled = non_fraudulent.sample(n=len(fraudulent), random_state=42)

# Concatenate the subsampled majority class with the minority class
balanced_dataset = pd.concat([fraudulent, non_fraudulent_sampled])

# Shuffle the dataset
balanced_dataset = balanced_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the balanced dataset to a new CSV file
balanced_file_path = 'balanced_dataset.csv'
balanced_dataset.to_csv(balanced_file_path, index=False)

# Display the balanced dataset
print(balanced_dataset)


overall_sample_size = 5

# Calculate the sample size for each class
fraudulent_sample_size = int(len(df[df['Class'] == 1]) / len(df) * overall_sample_size)
non_fraudulent_sample_size = overall_sample_size - fraudulent_sample_size

# Randomly select samples for each class
fraudulent_samples = df[df['Class'] == 1].sample(n=fraudulent_sample_size, random_state=42)
non_fraudulent_samples = df[df['Class'] == 0].sample(n=non_fraudulent_sample_size, random_state=42)

# Concatenate the samples
result_samples = pd.concat([fraudulent_samples, non_fraudulent_samples])

# Display the result samples
print(result_samples)

# Separate features and target variable
X = df.drop('Class', axis=1)
y = df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'RandomForestBalanced': BalancedRandomForestClassifier(random_state=42),
}

# Sampling techniques
sampling_techniques = {
    'RandomOversampling': RandomOverSampler(random_state=42),
    'RandomUndersampling': RandomUnderSampler(random_state=42),
    'SMOTE': SMOTE(random_state=42),
    'ADASYN': ADASYN(random_state=42),
    'BalancedSubsampling': None,  # No need for balancing, as the algorithm is inherently balanced
}

# Apply sampling techniques and evaluate models
for sampler_name, sampler in sampling_techniques.items():
    X_resampled, y_resampled = X_train, y_train  # Default, no resampling

    if sampler is not None:
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

    for model_name, model in models.items():
        # Train the model on resampled data
        model.fit(X_resampled, y_resampled)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{sampler_name} + {model_name} - Accuracy: {accuracy:.4f}")
