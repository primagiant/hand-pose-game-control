import pandas as pd
import numpy as np

# Load data CSV
file_path = "./model/dataset.csv"  # Ganti dengan path file CSV kamu

# Karena tidak ada header di CSV, kita definisikan header secara manual
columns = ["label"] + [f"x{i//2+1}" if i % 2 == 0 else f"y{i//2+1}" for i in range(42)]

df = pd.read_csv(file_path, header=None, names=columns)

# Pisahkan label dan koordinat
labels = df["label"].values
coordinates = df.iloc[:, 1:].values.reshape(-1, 21, 2)  # Ubah ke (n_samples, 21, 2)

# Cari nilai minimum x dan y di setiap baris
min_x = np.min(coordinates[:, :, 0], axis=1, keepdims=True)
min_y = np.min(coordinates[:, :, 1], axis=1, keepdims=True)

# Expand dims agar jadi (n_samples, 1, 1)
min_x = np.expand_dims(min_x, axis=2)
min_y = np.expand_dims(min_y, axis=2)

# Gabungkan min_x dan min_y ke bentuk (n_samples, 1, 2)
min_xy = np.concatenate([min_x, min_y], axis=2)  # Bentuk: (n_samples, 1, 2)

# Transformasi koordinat
transformed_coordinates = coordinates - min_xy

# Ubah kembali ke bentuk DataFrame
flattened_data = transformed_coordinates.reshape(-1, 42)
df_transformed = pd.DataFrame(flattened_data, columns=df.columns[1:])
df_transformed.insert(0, "label", labels)  # Masukkan kembali label di kolom pertama

# Simpan hasil ke CSV baru jika perlu
df_transformed.to_csv("./model/transformed_data.csv", index=False)

print("Data setelah transformasi:")
print(df_transformed.head())
