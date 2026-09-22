#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador didáctico de datos sintéticos: edad -> estatura

Este script crea un CSV con dos columnas: `age` (años) y `height` (centímetros).

Uso básico:
    python3 data/generate_height_data.py --n 500 --out data/estatura_ninos.csv --seed 42

Instalación de dependencias (pip):

    # crear y activar un virtualenv (recomendado)
    python3 -m venv .venv
    source .venv/bin/activate

    # instalar paquetes necesarios
    pip install --upgrade pip
    pip install pandas numpy

Salida:
    Archivo CSV con `n` filas y columnas `age`,`height`.
"""

import argparse
import os

import numpy as np
import pandas as pd


def generate(n: int, seed: int = 42) -> pd.DataFrame:
    """Genera un DataFrame con `n` muestras.

    Parámetros:
    - n: número de muestras a generar.
    - seed: semilla para el generador aleatorio (reproducibilidad).

    Retorna:
    - pd.DataFrame con columnas `age` y `height`.

    Detalles del modelo de generación (intencionalmente simple pero realista):
    - `ages` se muestrea uniformemente entre 0 y 18 años.
    - `heights` se calcula con una función base:
            height = 45 + 5.2 * age + 0.08 * age^2 + ruido
        donde el término cuadrático añade una ligera curvatura (crecimiento acelerado en fases),
        y `ruido` es gaussiano N(0, 3.0) para reflejar variaciones individuales.

    Notas didácticas:
    - La fórmula no pretende ser un modelo biomédico exacto, solo una relación plausible
        para enseñar ajuste supervisado y evaluación.
    - El ruido permite practicar métricas como MAE y observar scatter en predicciones.
    """
    rng = np.random.default_rng(seed)

    # 1) Generar edades en años, con dos decimales para simular mediciones reales
    ages = rng.uniform(0, 18, size=n)

    # 2) Generar estaturas con una relación no lineal + ruido
    #    - 45 cm como intercepto aproximado (recién nacido más bajo, sólo para ejemplo)
    #    - coeficiente lineal 5.2 (cm por año)
    #    - término cuadrático pequeño para introducir no linealidad
    #    - ruido gaussiano con std=3.0 cm para variabilidad
    heights = 45 + 5.2 * ages + 0.08 * (ages ** 2) + rng.normal(0, 3.0, size=n)

    # 3) Empaquetar y redondear para apariencia de datos reales
    df = pd.DataFrame({
        "age": np.round(ages, 2),
        "height": np.round(heights, 2),
    })
    return df


def main() -> None:
    """Interfaz de línea de comandos.

    Argumentos:
    --n: número de muestras (por defecto 500)
    --out: ruta de salida del CSV (por defecto data/estatura_ninos.csv)
    --seed: semilla aleatoria
    """
    p = argparse.ArgumentParser(description="Generador de datos sintéticos edad->estatura")
    p.add_argument("--n", type=int, default=500, help="Número de muestras")
    p.add_argument("--out", type=str, default="data/estatura_ninos.csv", help="Archivo CSV de salida")
    p.add_argument("--seed", type=int, default=42, help="Semilla aleatoria para reproducibilidad")
    args = p.parse_args()

    # Generar DataFrame
    df = generate(args.n, seed=args.seed)

    # Asegurarse de que la carpeta de salida exista
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Guardar CSV sin índice
    df.to_csv(args.out, index=False)
    print(f"Generado {len(df)} muestras en {args.out}")


if __name__ == "__main__":
    main()
