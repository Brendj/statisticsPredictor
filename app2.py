import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump

# Загрузка данных
df = pd.read_csv('dataset.csv')

# Предобработка данных
df = df.fillna(0)

label_encoder_sport = LabelEncoder()
label_encoder_position = LabelEncoder()
df['Вид спорта'] = label_encoder_sport.fit_transform(df['Вид спорта'])
df['Позиция'] = label_encoder_position.fit_transform(df['Позиция'].astype(str))

# Разделение данных на обучающую, валидационную и тестовую выборки
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Разделение данных на признаки и целевую переменную
X_train = train_df.drop('Коэффициент статистики', axis=1)
y_train = train_df['Коэффициент статистики']
X_val = val_df.drop('Коэффициент статистики', axis=1)
y_val = val_df['Коэффициент статистики']
X_test = test_df.drop('Коэффициент статистики', axis=1)
y_test = test_df['Коэффициент статистики']

# Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Создание модели
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))

# Компиляция модели
model.compile(loss='mean_squared_error', optimizer='adam')

# Обучение модели
callbacks = [EarlyStopping(patience=10), ModelCheckpoint('my_model.keras', save_best_only=True)]
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=callbacks)

# Оценка модели
loss = model.evaluate(X_test, y_test)
print(f'Loss: {loss}')

# Сохранение модели и объектов предварительной обработки
dump({'model': model, 'scaler': scaler, 'label_encoder_sport': label_encoder_sport, 'label_encoder_position': label_encoder_position}, 'model_and_preprocessing_objects.joblib')

# Выбор 5 случайных строк из датасета
random_samples = df.sample(5)

# Предобработка выбранных данных
random_samples_features = random_samples.drop('Коэффициент статистики', axis=1)
random_samples_features = scaler.transform(random_samples_features)

# Предсказание модели
predictions = model.predict(random_samples_features)

# Вывод входных данных и предсказаний
for i in range(5):
    print("Входные данные:")
    print(random_samples.iloc[i])
    print("Предсказание модели:", predictions[i])
    print(" ")
