import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

import kerastuner as kt

#############################################
# Função para criar sequências temporais a partir do CSV
#############################################
def create_sequences(csv_path, sequence_length=30):
    df = pd.read_csv(csv_path)
    df = df.sort_values(by=['video', 'frame'])
    use_label = 'label' in df.columns
    feature_cols = [col for col in df.columns if col not in ['video', 'frame', 'label']]
    data = df[feature_cols].values

    sequences = []
    labels = []
    for i in range(len(data) - sequence_length + 1):
        seq = data[i:i+sequence_length]
        sequences.append(seq)
        if use_label:
            labels.append(df.iloc[i]['label'])
    X = np.array(sequences)
    y = np.array(labels) if use_label else None
    return X, y

#############################################
# Função para construir o modelo com hiperparâmetros variáveis
#############################################
def build_model_hp(hp, timesteps, feature_dim):
    model = Sequential()
    units = hp.Choice('units', values=[32, 64, 128])
    dropout_rate = hp.Float('dropout', 0.0, 0.5, step=0.1)

    model.add(GRU(units,
                  input_shape=(timesteps, feature_dim),
                  activation='tanh',
                  recurrent_activation='sigmoid',
                  recurrent_dropout=dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    learning_rate = hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='log')
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

#############################################
# Função para treinar e avaliar o modelo com Keras Tuner
#############################################
def tune_model(X, y, sequence_length, feature_dim, max_trials=10, epochs=10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    tuner = kt.RandomSearch(
        lambda hp: build_model_hp(hp, timesteps=sequence_length, feature_dim=feature_dim),
        objective='val_accuracy',
        max_trials=max_trials,
        executions_per_trial=1,
        directory='gru_tuner_dir',
        project_name='gru_model_tuning'
    )
    
    tuner.search(X_train, y_train, epochs=epochs, validation_split=0.2)
    
    tuner.results_summary()
    
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print("\nMelhores hiperparâmetros encontrados:")
    print(best_hp.values)
    
    test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    y_pred_prob = best_model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc:.4f}")
    
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.show()
    
    return best_model

#############################################
# Função principal (Treinamento isolado)
#############################################
def main():
    # Caminho para o CSV mesclado (ajuste para o seu ambiente)
    merged_csv_path = r'C:\Users\vitor\Downloads\Projeto-IA\Dados\dados_mesclados.csv'
    sequence_length = 30

    X, y = create_sequences(merged_csv_path, sequence_length=sequence_length)
    print("Shape dos dados (X):", X.shape)
    print("Shape dos rótulos (y):", y.shape)

    feature_dim = X.shape[2]
    print("Feature dimension:", feature_dim)

    best_model = tune_model(X, y, sequence_length, feature_dim, max_trials=10, epochs=10)

    # Salva o modelo treinado para ser utilizado depois pela interface
    best_model.save("best_model.keras")
    print("Modelo treinado salvo")

if __name__ == '__main__':
    main()