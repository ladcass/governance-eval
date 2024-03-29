import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical

# Load the dataset
df = pd.read_csv('data_pull/db/full_v0.csv')
df = df.dropna()

# Encode the 'roa' classes
label_encoder_roa = LabelEncoder()
df['roa_transformed'] = label_encoder_roa.fit_transform(df['roachange'])

# One-hot encode 'ticker'
ohe_ticker = OneHotEncoder()
ticker_encoded = ohe_ticker.fit_transform(df[['ticker']]).toarray()

# Define binary and numeric columns
numeric_cols = ['uniscoreavg', 'tenure', 'female_pct', 'shannon', 'num_committees', 'dirs', 'over75_pct',
                'num_meetings', 'avg_meetings']
binary_cols = ['bachelors', 'masters', 'mba', 'phd', 'founder', 'internal_promotion', 'is_co_ceo', 'diversity_board']

# Standardize 'uniscoreavg' and 'tenure'
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Convert binary columns to numeric
df[binary_cols] = df[binary_cols].astype(int)

# Build and evaluate model for entire dataset
X = df[numeric_cols + binary_cols]
y = df['roa_transformed']

scaler = StandardScaler()
X = scaler.fit_transform(X)

y = to_categorical(y)
num_classes = y.shape[1]


# Define the CNNs to then be called in the dictionary
def build_model_1(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_model_2(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(128, activation='tanh'),
        BatchNormalization(),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_model_3(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_model_4(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='tanh'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_model_5(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Define models in a dictionary
models = {
    "rf_model_1": lambda input_shape, num_classes: RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    "rf_model_2": lambda input_shape, num_classes: RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42),
    "rf_model_3": lambda input_shape, num_classes: RandomForestClassifier(n_estimators=100, min_samples_split=5, class_weight='balanced', random_state=42),
    "gbm_model_1": lambda input_shape, num_classes: GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
    "gbm_model_2": lambda input_shape, num_classes: GradientBoostingClassifier(n_estimators=100, subsample=0.8, random_state=42),
    "gbm_model_3": lambda input_shape, num_classes: GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    "svm_model_1": lambda input_shape, num_classes: SVC(kernel='linear', class_weight='balanced', random_state=42),
    "svm_model_2": lambda input_shape, num_classes: SVC(C=0.5, kernel='rbf', class_weight='balanced', random_state=42),
    "svm_model_3": lambda input_shape, num_classes: SVC(gamma=0.01, kernel='rbf', class_weight='balanced', random_state=42),
    "xgb_model_1": lambda input_shape, num_classes: XGBClassifier(max_depth=8, scale_pos_weight=1, use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "xgb_model_2": lambda input_shape, num_classes: XGBClassifier(learning_rate=0.05, scale_pos_weight=1, use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "xgb_model_3": lambda input_shape, num_classes: XGBClassifier(min_child_weight=5, scale_pos_weight=1, use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "logreg_model_1": lambda input_shape, num_classes: LogisticRegression(C=0.5, class_weight='balanced', random_state=42, max_iter=1000),
    "logreg_model_2": lambda input_shape, num_classes: LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42, max_iter=1000),
    "logreg_model_3": lambda input_shape, num_classes: LogisticRegression(class_weight={0: 1, 1: 2, 2: 3}, random_state=42, max_iter=1000)
}

cnn_models = {
    "cnn_model_1": build_model_1,
    "cnn_model_2": build_model_2,
    "cnn_model_3": build_model_3,
    "cnn_model_4": build_model_4,
    "cnn_model_5": build_model_5
}


# Function to run model multiple times and record accuracies
def run_model_multiple_times(model_func, X, y, num_runs=1000):
    accuracies = []
    for _ in range(num_runs):
        X_train, X_test, y_train, y_test, tickers_train, tickers_test = train_test_split(X, y, df['ticker'],
                                                                                         test_size=0.2, random_state=42)
        model = model_func(X_train.shape[1:], num_classes)
        model.fit(X_train, np.argmax(y_train, axis=1))
        accuracy = model.score(X_test, np.argmax(y_test, axis=1))
        accuracies.append(accuracy)
    return accuracies


def run_cnn_model_multiple_times(model_func, X, y, num_runs=100):
    cnn_accuracies = []
    count = 0
    while count < num_runs:
        try:
            # Splitting data for each iteration to ensure different splits
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model = model_func(X_train.shape[1:], num_classes)
            model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1)
            _, accuracy = model.evaluate(X_test, y_test, verbose=0)
            cnn_accuracies.append(accuracy)
            count += 1
        except Exception as e:
            None
    return cnn_accuracies


# Store results
results = []

for model_name, model_func in models.items():
    print(f"Running {model_name}")
    accuracies = run_model_multiple_times(model_func, X, y)
    results.append((model_name, accuracies))

# Write results to a .txt file
with open("models/accuracies/model_accuracies.txt", "w") as f:
    for model_name, accuracies in results:
        accuracies_str = ", ".join([str(acc) for acc in accuracies])
        f.write(f"{model_name}: [{accuracies_str}]\n")

for cnn_model_name, cnn_model_func in cnn_models.items():
    results = []
    print(f"Running {cnn_model_name}")
    accuracies = run_cnn_model_multiple_times(cnn_model_func, X, y)
    results.append((cnn_model_name, accuracies))
    with open(f"models/accuracies/{cnn_model_name}_accuracies.txt", "w") as f:
        accuracies_str = ", ".join([str(acc) for acc in accuracies])
        f.write(f"{cnn_model_name}: [{accuracies_str}]\n")
