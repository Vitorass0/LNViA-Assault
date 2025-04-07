import os
import streamlit as st
import numpy as np
from keras.models import load_model
from processamento import process_video, create_sequences
import yt_dlp


def baixar_video_youtube(url, caminho_saida="videos"):
    try:
        if not os.path.exists(caminho_saida):
            os.makedirs(caminho_saida)

        nome_arquivo = os.path.join(caminho_saida, "%(title)s.%(ext)s")

        ydl_opts = {
            'format': 'mp4',
            'outtmpl': nome_arquivo,
            'quiet': True,
            'noplaylist': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info_dict)
            return video_path

    except Exception as e:
        st.error(f"❌ Erro no download com yt-dlp: {e}")
        return None


def carregar_modelo():
    try:
        caminho_modelo = "best_model.keras"
        if not os.path.exists(caminho_modelo):
            caminho_modelo = "best_model.h5"  # fallback
        return load_model(caminho_modelo, compile=False)
    except Exception as e:
        st.error(f"❌ Erro ao carregar o modelo: {e}")
        return None


def classificar_video(video_path, modelo):
    try:
        csv_temp = os.path.splitext(os.path.basename(video_path))[0] + "_temp.csv"
        process_video(video_path, csv_temp, frame_rate=1)

        X, _ = create_sequences(csv_temp, sequence_length=30)

        if X is None or len(X) == 0:
            st.warning("⚠️ Não foi possível extrair sequências do vídeo.")
            return

        pred = modelo.predict(X)
        media_pred = np.mean(pred)
        resultado = "🚨 Agressão detectada!" if media_pred > 0.4 else "✅ Comportamento normal."
        st.subheader(resultado)
        st.text(f"Confiança média: {media_pred:.2f}")

    except Exception as e:
        st.error(f"❌ Erro ao processar o vídeo: {e}")
    finally:
        if os.path.exists(csv_temp):
            os.remove(csv_temp)


def main():
    st.title("Classificador de Vídeos - Agressão ou Normal")

    modelo = carregar_modelo()
    if modelo is None:
        return

    opcao = st.radio("Escolha a forma de entrada do vídeo:", ("Upload", "YouTube"))

    caminho_video = None

    if opcao == "Upload":
        video_file = st.file_uploader("Envie um vídeo", type=["mp4"])
        if video_file is not None:
            temp_path = os.path.join("temp", video_file.name)
            os.makedirs("temp", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(video_file.read())
            caminho_video = temp_path

    elif opcao == "YouTube":
        url = st.text_input("Cole o link do vídeo do YouTube:")
        if url:
            caminho_video = baixar_video_youtube(url)

    if caminho_video and os.path.exists(caminho_video):
        st.video(caminho_video)
        st.write("Processando vídeo...")
        classificar_video(caminho_video, modelo)


if __name__ == "__main__":
    main()
