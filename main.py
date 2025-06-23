import pandas as pd
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from transformers import BertTokenizer, TFBertForSequenceClassification

# Abrir o arquivo de saída para salvar os resultados
output = []

# 1. Carregar dados
df = pd.read_csv("data/fake_or_real_news.csv")
df = df[['text', 'label']].dropna()
df['label'] = df['label'].map({'REAL': 0, 'FAKE': 1})

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42)

# 2. Random Forest
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

rf = RandomForestClassifier()

start = time.time()
rf.fit(X_train_tfidf, y_train)
end = time.time()
output.append(f"\nTempo de treinamento (Random Forest): {end - start:.2f} segundos")

start = time.time()
rf_preds = rf.predict(X_test_tfidf)
end = time.time()
output.append(f"Tempo de inferência (Random Forest): {end - start:.2f} segundos")

rf_report = classification_report(y_test, rf_preds)
output.append("Random Forest Results:\n" + rf_report)

# 3. LSTM
tokenizer_lstm = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
tokenizer_lstm.fit_on_texts(X_train)
X_train_seq = tokenizer_lstm.texts_to_sequences(X_train)
X_test_seq = tokenizer_lstm.texts_to_sequences(X_test)
X_train_pad = tf.keras.preprocessing.sequence.pad_sequences(X_train_seq, maxlen=500)
X_test_pad = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq, maxlen=500)

lstm_model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=500),
    LSTM(64),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

start = time.time()
lstm_model.fit(X_train_pad, y_train, epochs=2, batch_size=64, validation_split=0.1)
end = time.time()
output.append(f"\nTempo de treinamento (LSTM): {end - start:.2f} segundos")

start = time.time()
lstm_preds = (lstm_model.predict(X_test_pad) > 0.5).astype("int32")
end = time.time()
output.append(f"Tempo de inferência (LSTM): {end - start:.2f} segundos")

lstm_report = classification_report(y_test, lstm_preds)
output.append("LSTM Results:\n" + lstm_report)

# 4. Fine-tuning BERT

# 4.1 Tokenizer e função de tokenização
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def encode_texts(texts, max_len=128):
    return bert_tokenizer(
        list(texts),
        max_length=max_len,
        truncation=True,
        padding='max_length',
        return_tensors='tf'
    )

train_encodings = encode_texts(X_train)
test_encodings = encode_texts(X_test)

train_labels = tf.convert_to_tensor(y_train.values)
test_labels = tf.convert_to_tensor(y_test.values)

train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels)).shuffle(1000).batch(16)
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), test_labels)).batch(16)

# 4.2 Modelo BERT para classificação
bert_model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")

optimizer = tf.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = ['accuracy']
bert_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# 4.3 Treinamento
start = time.time()
bert_model.fit(train_dataset, epochs=2, validation_data=test_dataset)
end = time.time()
output.append(f"\nTempo de treinamento (BERT): {end - start:.2f} segundos")

# 4.4 Inferência
start = time.time()
preds = bert_model.predict(test_dataset)
logits = preds.logits
predicted_classes = tf.argmax(logits, axis=1)
end = time.time()
output.append(f"Tempo de inferência (BERT): {end - start:.2f} segundos")

bert_report = classification_report(y_test, predicted_classes.numpy())
output.append("BERT Results:\n" + bert_report)

# Salvar todos os resultados em um arquivo
with open("resultados.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output))

print("\n✅ Resultados salvos em 'resultados.txt'")
