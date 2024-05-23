import pandas as pd
import os

# Definindo os caminhos para os dados
raw_data_path = os.path.join('data', 'raw', 'raw_data.csv')
processed_data_path = os.path.join('data', 'processed', 'processed_data.csv')

def load_raw_data(path):
    return pd.read_csv(path)

def process_data(df):
    df = df.dropna()
    df = (df - df.mean()) / df.std()
    return df

def save_processed_data(df, path):
    df.to_csv(path, index=False)

if __name__ == "__main__":
    raw_data = load_raw_data(raw_data_path)
    processed_data = process_data(raw_data)
    save_processed_data(processed_data, processed_data_path)
    print(f'Dados processados salvos em: {processed_data_path}')
