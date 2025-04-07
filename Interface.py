import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
import tempfile
import torch
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from pytube import YouTube  # Biblioteca para baixar vídeos do YouTube

# --- Função para extrair poses do vídeo ---
def extract_keypoints_from_video(video_path, frame_rate=5, sequence_length=30):
    model = YOLO("yolov8n-pose.pt")
    cap = cv2.VideoCapture(video_path)
    data = []
    expected_kp_len = 51  # 17 keypoints * (x, y, conf)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 360))
        if frame_idx % frame_rate == 0:
            results = model(frame)
            if results and results[0].keypoints is not None and results[0].keypoints.data.shape[0] > 0:
                kp = results[0].keypoints.data[0]
                if torch.isnan(kp).any():
                    kp = torch.nan_to_num(kp, nan=0.0)
                kp_flat = kp.flatten().tolist()
                if len(kp_flat) < expected_kp_len:
                    kp_flat += [0] * (expected_kp_len - len(kp_flat))
                elif len(kp_flat) > expected_kp_len:
                    kp_flat = kp_flat[:expected_kp_len]
                data.append(kp_flat)
        frame_idx += 1

    cap.release()

    # Verifica se há frames suficientes para sequência
    if len(data) < sequence_length:
        st.warning("⚠️ Vídeo muito curto. Adicionando padding...")
        while len(data) < sequence_length:
            data.append([0] * expected_kp_len)

    # Corta ou mantém o necessário
    data = data[:sequence_length]
    return np.expand_dims(np.array(data), axis=0)  # (1, 30, 51)

# --- Função para baixar vídeo do YouTube ---
def download_youtube_video(youtube_url):
    try:
        yt = YouTube(youtube_url)
        stream = yt.streams.filter(file_extension="mp4", res="360p").first()
        if not stream:
            stream = yt.streams.get_lowest_resolution()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        stream.download(output_path=os.path.dirname(temp_file.name), filename=os.path.basename(temp_file.name))
        return temp_file.name
    except Exception as e:
        st.error(f"Erro ao baixar o vídeo do YouTube: {e}")
        return None

# --- Streamlit Interface ---
st.set_page_config(page_title="Detector de Brigas em Vídeos", layout="centered")
st.title("👁️‍🗨️ Classificador de Brigas em Vídeos com YOLO + LSTM")

# Entrada para link do YouTube
youtube_url = st.text_input("📥 Insira o link do vídeo do YouTube:")

if youtube_url:
    st.write("🔽 Baixando o vídeo do YouTube...")
    video_path = download_youtube_video(youtube_url)

    if video_path:
        st.video(video_path)
        st.write("🔍 Extraindo características...")

        try:
            X = extract_keypoints_from_video(video_path)

            # Normaliza os dados
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

            # Carrega modelo treinado
            model = load_model("modelo_lstm.h5")

            # Faz a previsão
            prediction = model.predict(X)[0][0]
            label = "🚨 Briga Detectada" if prediction >= 0.5 else "✅ Comportamento Normal"
            st.subheader(f"Resultado: {label}")
            st.write(f"Confiabilidade: {prediction:.2%}")

        except Exception as e:
            st.error(f"Erro ao processar o vídeo: {e}")

        # Remove arquivo temporário
        os.remove(video_path)