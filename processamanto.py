import os
import cv2
import pandas as pd
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm

def process_video(video_path, output_csv, frame_rate=1, label=None):
    model = YOLO("yolov8n-pose.pt")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    data = []
    frame_idx = 0
    expected_kp_len = 51

    for _ in tqdm(range(total_frames), desc=f"Processando {os.path.basename(video_path)}"):
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
                kp_flat = (kp_flat + [0] * expected_kp_len)[:expected_kp_len]
                row = [os.path.basename(video_path), frame_idx] + ([label] if label is not None else []) + kp_flat
                data.append(row)
            else:
                row = [os.path.basename(video_path), frame_idx] + ([label] if label is not None else []) + [0]*expected_kp_len
                data.append(row)
        frame_idx += 1
    cap.release()

    base_cols = ['video', 'frame'] + (['label'] if label is not None else [])
    kp_cols = [f'kp{i}_{dim}' for i in range(17) for dim in ['x', 'y', 'conf']]
    df = pd.DataFrame(data, columns=base_cols + kp_cols)
    df.fillna(0, inplace=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ CSV salvo: {output_csv}")

def gerar_csvs_e_mesclar(dir_assalto, dir_nao_assalto, frame_rate=5):

    csv_files = []

    def is_video_file(filename):
        return filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

    for file in os.listdir(dir_assalto):
        if is_video_file(file):
            path = os.path.join(dir_assalto, file)
            nome_csv = os.path.splitext(file)[0] + ".csv"
            process_video(path, nome_csv, frame_rate, label=1)
            csv_files.append(nome_csv)

    for file in os.listdir(dir_nao_assalto):
        if is_video_file(file):
            path = os.path.join(dir_nao_assalto, file)
            nome_csv = os.path.splitext(file)[0] + ".csv"
            process_video(path, nome_csv, frame_rate, label=0)
            csv_files.append(nome_csv)

    df_list = []
    for csv in csv_files:
        if os.path.exists(csv):
            df = pd.read_csv(csv)
            df.fillna(0, inplace=True)
            df_list.append(df)

    if df_list:
        df_merged = pd.concat(df_list, ignore_index=True)
        df_merged.fillna(0, inplace=True)
        df_merged.to_csv("dados_mesclados.csv", index=False)
        print("✅ CSV mesclado salvo: dados_mesclados.csv")
    else:
        print("❌ Nenhum CSV válido gerado.")
    
# processamento.py
import pandas as pd
import numpy as np

def create_sequences(csv_path, sequence_length=30):
    df = pd.read_csv(csv_path)
    if df.isnull().values.any():
        print("❗ NaN detectado no CSV. Substituindo por 0.")
        df.fillna(0, inplace=True)
    df = df.sort_values(by=['video', 'frame'])

    use_label = 'label' in df.columns
    feature_cols = [col for col in df.columns if col not in ['video', 'frame', 'label']]
    data = df[feature_cols].values

    sequences, labels = [], []
    for i in range(len(data) - sequence_length + 1):
        seq = data[i:i+sequence_length]
        sequences.append(seq)
        if use_label:
            labels.append(df.iloc[i]['label'])

    X = np.array(sequences)
    y = np.array(labels) if use_label else None
    return X, y


if __name__ == "__main__":
    dir_assalto = r'C:\Users\vitor\Downloads\Assault'
    dir_nao_assalto = r'C:\Users\vitor\Downloads\Normal_Videos_for_Event_Recognition'

    gerar_csvs_e_mesclar(dir_assalto, dir_nao_assalto)
