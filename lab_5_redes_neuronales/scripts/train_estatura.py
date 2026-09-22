#!/usr/bin/env python3
"""Entrenamiento simple en PyTorch para predecir estatura a partir de la edad.

Este archivo contiene un ejemplo mínimo y comentado para uso didáctico en clase.
Incluye:
- Carga de datos desde CSV
- Normalización de la característica de entrada (edad)
- División entrenamiento/prueba
- Definición de una red pequeña (`SimpleNet`)
- Bucle de entrenamiento y evaluación periódica (MSE, MAE)
- Guardado del estado del modelo y estadísticas necesarias para inferencia

Instalación de dependencias (pip) — Linux / macOS:

    # 1) Crear y activar un virtualenv (recomendado)
    python3 -m venv .venv
    source .venv/bin/activate

    # 2) Actualizar pip e instalar paquetes
    pip install --upgrade pip
    pip install pandas numpy torch

Instalación de dependencias (pip) — Windows (PowerShell):

    # 1) Crear y activar un virtualenv
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

    # 2) Actualizar pip e instalar paquetes
    pip install --upgrade pip
    pip install pandas numpy torch

Alternativa con requirements.txt:

    pip install -r requirements.txt

Notas pedagógicas:
- Para problemas reales, se deben revisar más características (sexo, peso, historial),
    más datos, normalización adecuada según conjunto de entrenamiento, y validación cruzada.
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split


class SimpleNet(nn.Module):
    """Red neuronal simple para regresión.

    Arquitectura:
    - Capa lineal (1 -> 16)
    - ReLU
    - Capa lineal (16 -> 8)
    - ReLU
    - Capa lineal (8 -> 1)

        Observaciones didácticas:
        - La entrada es una sola característica (`age`) y la salida es una única predicción (`height`).
        - Capas pequeñas para aprendizaje rápido en CPU y evitar overfitting en datos sintéticos.

        Funciones de activación (rationale):
        - Capas ocultas: usamos `ReLU` (Rectified Linear Unit).
            * Ventajas: es computacionalmente eficiente, introduce no saturación para
                entradas positivas (evita gradientes muy pequeños) y suele acelerar
                la convergencia en redes feed-forward. Además produce activaciones
                dispersas (sparse), lo que puede mejorar generalización en redes simples.
            * Riesgos: unidades "muertas" (si la neurona siempre produce valores
                negativos) — solución posible: `LeakyReLU` o `ELU` si ocurre.
            * Alternativas y cuándo usarlas:
                - `tanh` o `sigmoid`: antes eran comunes, pero se saturan para valores
                    grandes y producen gradientes pequeños; útiles si la salida interna
                    debe estar acotada o para arquitecturas específicas.
                - `LeakyReLU` / `ParametricReLU` / `ELU`: variantes que mitigan el
                    problema de unidades muertas manteniendo ventajas similares a ReLU.

        - Capa de salida: no usamos activación (lineal).
            * Razonamiento: en problemas de regresión predictiva (altura en cm) se
                requiere una salida continua y no acotada; una función lineal permite
                predecir cualquier valor real. Si la variable objetivo estuviera
                acotada, podría considerarse una transformación o una activación
                apropiada en la salida.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagación hacia adelante.

        x: tensor de forma (batch_size, 1)
        retorna: tensor de forma (batch_size, 1)
        """
        return self.net(x)


def load_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Carga el CSV y devuelve matrices numpy para X (edad) e y (estatura).

    - Aseguramos que los tipos sean `float32` para compatibilidad con PyTorch.
    - `X` se devuelve con forma (n_samples, 1) y `y` con forma (n_samples, 1).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Archivo de datos no encontrado: {p} \nPasa --data con la ruta correcta o ejecuta el script desde la raíz del proyecto.")
    df = pd.read_csv(p)
    x = df[["age"]].values.astype(np.float32)
    y = df[["height"]].values.astype(np.float32)
    return x, y


def main() -> None:
    """Función principal con argumentos de línea de comandos.

    Argumentos relevantes:
    --data: ruta al CSV con columnas `age`,`height`.
    --epochs: número de épocas de entrenamiento.
    --batch: tamaño de batch.
    --lr: tasa de aprendizaje para el optimizador Adam.
    --out: ruta de guardado del modelo (estado + normalización).
    """
    p = argparse.ArgumentParser(description="Entrena una red simple edad->estatura con PyTorch")
    # Valores por defecto relativos a la raíz del repositorio (dos niveles sobre este archivo)
    repo_root = Path(__file__).resolve().parent.parent
    default_data = str(repo_root / "data" / "estatura_ninos.csv")
    default_out = str(repo_root / "models" / "estatura_model.pth")
    p.add_argument("--data", type=str, default=default_data)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", type=str, default=default_out)
    args = p.parse_args()

    # 1) Cargar datos
    x, y = load_data(args.data)

    # 2) Normalizar la entrada: importante para la convergencia del optimizador.
    #
    # Explicación didáctica (normalización z-score):
    # - Aquí usamos normalización por media y desviación estándar (z-score):
    #       x_norm = (x - mean) / std
    #   Esto centra los datos en media 0 y desviación estándar 1.
    # - ¿Por qué hacerlo?
    #   * Mejora la estabilidad y velocidad de convergencia del optimizador (Adam/SGD),
    #     porque las entradas están en una escala similar y las capas lineales no
    #     reciben valores con magnitudes muy distintas.
    #   * Evita que las activaciones se saturen en funciones no lineales (ReLU/ tanh),
    #     reduciendo problemas de gradientes muy pequeños o grandes.
    # - Epsilon: se añade un valor muy pequeño (`+ 1e-8`) para evitar división por cero
    #   en el caso (hipotético) de desviación estándar nula.
    # - Alternativas:
    #   * Min-max scaling: (x - min) / (max - min) → útil cuando queremos mantener
    #     valores en un rango [0,1], pero es sensible a outliers.
    #   * Robust scaling: usar la mediana y el IQR para ser robusto a outliers.
    # - Guardado de `mean` y `std`:
    #   Es crucial almacenar estos parámetros calculados sobre el conjunto de entrenamiento
    #   y reutilizarlos en inferencia (o en nuevos datos) para asegurar que la entrada
    #   se normaliza de la misma forma que durante el entrenamiento.
    x_mean, x_std = x.mean(axis=0), x.std(axis=0) + 1e-8
    x_norm = (x - x_mean) / x_std

    # Convertir a tensores de PyTorch (float32)
    X = torch.from_numpy(x_norm)
    Y = torch.from_numpy(y)

    # 3) Preparar dataset y dividir en entrenamiento/prueba (80/20)
    dataset = TensorDataset(X, Y)
    n_test = int(0.2 * len(dataset))
    n_train = len(dataset) - n_test
    train_set, test_set = random_split(dataset, [n_train, n_test])

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch)

    # 4) Configurar dispositivo — FORZAR CPU
    # En este entorno docente nunca usamos CUDA (GPU). Forzamos `cpu` para:
    # - Evitar dependencias a hardware específico en la clase
    # - Asegurar reproducibilidad entre equipos
    # - Evitar errores si CUDA/driver no está instalado
    device = torch.device("cpu")
    model = SimpleNet().to(device)

    # 5) Optimizador y función de pérdida
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Elección de la función de pérdida:
    # - MSE (Mean Squared Error, Error Cuadrático Medio): penaliza más los errores grandes
    #   porque eleva al cuadrado la diferencia (y_true - y_pred)^2. Es útil cuando
    #   queremos castigar fuertemente grandes desviaciones y su derivada es suave
    #   (conviene para optimizadores que usan gradientes).
    # - MAE (Mean Absolute Error, Error Absoluto Medio): mide la magnitud media del
    #   error sin elevar al cuadrado, por lo que es más robusto frente a outliers
    #   (los errores grandes no se amplifican tanto). Su derivada es discontinua
    #   en 0, lo que puede afectar la convergencia en algunos casos.
    # En este ejemplo usamos MSE como criterio principal (más habitual en regresión
    # con redes neuronales) pero también reportamos MAE en la evaluación porque
    # proporciona una medida fácil de interpretar (cm promedio de error) y ayuda a
    # detectar si el modelo comete errores puntuales grandes.
    criterion = nn.MSELoss()

    # 6) Bucle de entrenamiento
    #    - Calculamos la pérdida MSE sobre el batch, backprop y actualizamos parámetros.
    #    - Almacenar promedio ponderado por número de muestras para reporting.
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # loss.item() es la media por elemento del batch; multiplicamos por tamaño del batch
            running += loss.item() * xb.size(0)
        train_loss = running / n_train

        # 7) Evaluación periódica: calculamos MSE y MAE en el conjunto de prueba
        if epoch % 50 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                test_losses = []
                maes = []
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    # MSE por batch ponderado
                    test_losses.append(criterion(pred, yb).item() * xb.size(0))
                    # MAE por batch ponderado (promedio absoluto)
                    maes.append(torch.abs(pred - yb).mean().item() * xb.size(0))
                test_loss = sum(test_losses) / n_test
                mae = sum(maes) / n_test
            print(f"Epoch {epoch:03d}  Train MSE: {train_loss:.4f}  Test MSE: {test_loss:.4f}  Test MAE: {mae:.4f}")

    # 8) Guardar el modelo y la normalización (necesarios para inferencia posterior)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
    }, args.out)
    print(f"Modelo guardado en {args.out}")


if __name__ == "__main__":
    main()
