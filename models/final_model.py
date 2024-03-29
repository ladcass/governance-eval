import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential([
    Input(shape=X_train.shape[1:]),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(128, activation='tanh'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(y.shape[1], activation='softmax')
])
model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1)

# Evaluate the model and print the accuracy
_, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Make predictions on whole dataset and save to CSV: final.csv
predictions = model.predict(X)
df['roapredicted'] = label_encoder_roa.inverse_transform(predictions.argmax(axis=1))
df_pred = df[['ticker', 'roapredicted']]

# Save predictions onto the original dataset given that the one previously loaded has been transformed
df = pd.read_csv('data_pull/db/full_v0.csv')
df = df.dropna()

df = df.merge(df_pred, on='ticker', how='left')

df.drop(columns=['roachange']).to_csv('data_pull/db/final.csv', index=False)
