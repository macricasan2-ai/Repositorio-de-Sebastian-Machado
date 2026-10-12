"""
================================================================================
  EXPERIMENTOS DE HIPERPARÁMETROS - RED NEURONAL (RED WINE QUALITY)
================================================================================

Entrena la misma red de red_neuronal_vino.py (entrada -> 16 -> 8 -> 1, Adam, MSE)
con distintos valores de tres hiperparámetros, cambiando UNO a la vez:

  - Épocas:              30  /  100 (base)  /  300
  - Tasa de aprendizaje: 0.0001  /  0.001 (base)  /  0.01  /  0.1
  - Tamaño de lote:      16  /  64 (base)  /  256

¿POR QUÉ 3 SEMILLAS?
  Una sola corrida depende de la inicialización aleatoria de los pesos. Cada
  configuración se entrena N_SEMILLAS veces y se promedia, así sabes si una
  diferencia es real o solo ruido (la desviación estándar aparece en la tabla).

GENERA (en esta misma carpeta):
  - resultados_experimentos.csv        tabla con las métricas de cada configuración
  - comparacion_hiperparametros.png    curvas de validación y entrenamiento (+ .pdf)
  No toca historial_entrenamiento.csv ni curva_aprendizaje del script anterior.

REQUISITOS:
  pip install torch pandas scikit-learn matplotlib
  (el CSV del vino debe estar en la misma carpeta, igual que con red_neuronal_vino.py)
================================================================================
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# "Agg" guarda las gráficas en archivo sin abrir ventanas (evita problemas en terminal).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==============================================================================
# SECCIÓN 1: CONFIGURACIÓN
# ==============================================================================

CARPETA_SCRIPT = Path(__file__).resolve().parent

# None = busca automáticamente un .csv con "wine" en el nombre (igual que antes).
ARCHIVO_CSV = None
ARCHIVO_RESULTADOS = CARPETA_SCRIPT / "resultados_experimentos.csv"
ARCHIVO_COMPARACION = CARPETA_SCRIPT / "comparacion_hiperparametros"  # .png y .pdf

COLUMNA_OBJETIVO = "quality"
COLUMNAS_IGNORAR = ["Id", "id"]

PORCENTAJE_PRUEBA = 0.2
PORCENTAJE_VALIDACION = 0.2
SEMILLA_DATOS = 42        # misma división de datos que red_neuronal_vino.py
N_SEMILLAS = 3            # cuántas veces se entrena cada configuración
ESCALAR_OBJETIVO = True   # pérdida 1.0 = predecir siempre el promedio

# Valores base (los mismos de red_neuronal_vino.py)
BASE = {"epocas": 100, "lr": 0.001, "lote": 64}

# Cada familia cambia UN hiperparámetro; {} significa "usar el valor base".
# Puedes agregar o quitar valores libremente.
EXPERIMENTOS = {
    "Épocas": [
        ("30", {"epocas": 30}),
        ("100 (base)", {}),
        ("300", {"epocas": 300}),
    ],
    "Tasa de aprendizaje": [
        ("0.0001", {"lr": 0.0001}),
        ("0.001 (base)", {}),
        ("0.01", {"lr": 0.01}),
        ("0.1", {"lr": 0.1}),
    ],
    "Tamaño de lote": [
        ("16", {"lote": 16}),
        ("64 (base)", {}),
        ("256", {"lote": 256}),
    ],
}

if ARCHIVO_CSV is None:
    candidatos = sorted(p for p in CARPETA_SCRIPT.glob("*.csv") if "wine" in p.name.lower())
    if not candidatos:
        raise FileNotFoundError(
            "No encontré ningún .csv con 'wine' en el nombre en esta carpeta. "
            "Ponlo junto al script o escribe su nombre en ARCHIVO_CSV."
        )
    ARCHIVO_CSV = candidatos[0]


# ==============================================================================
# SECCIÓN 2: CARGAR Y PREPARAR LOS DATOS (igual que red_neuronal_vino.py)
# ==============================================================================

print("=" * 60)
print("  CARGANDO DATOS...")
print("=" * 60)

with open(ARCHIVO_CSV, encoding="utf-8-sig") as f:
    primera_linea = f.readline()
separador = ";" if primera_linea.count(";") > primera_linea.count(",") else ","

datos = pd.read_csv(ARCHIVO_CSV, sep=separador)
datos.columns = datos.columns.str.strip()

if COLUMNA_OBJETIVO not in datos.columns:
    raise ValueError(
        f"No existe la columna '{COLUMNA_OBJETIVO}'. Columnas disponibles: {list(datos.columns)}"
    )

datos = datos.drop(columns=[c for c in COLUMNAS_IGNORAR if c in datos.columns]).dropna()
COLUMNAS_ENTRADA = [c for c in datos.columns if c != COLUMNA_OBJETIVO]

X = datos[COLUMNAS_ENTRADA].to_numpy(dtype=float)
y = datos[COLUMNA_OBJETIVO].to_numpy(dtype=float)

X_ent_val, X_prueba, y_ent_val, y_prueba = train_test_split(
    X, y, test_size=PORCENTAJE_PRUEBA, random_state=SEMILLA_DATOS
)
X_entren, X_validacion, y_entren, y_validacion = train_test_split(
    X_ent_val, y_ent_val, test_size=PORCENTAJE_VALIDACION, random_state=SEMILLA_DATOS
)

escalador = StandardScaler()
X_entren = escalador.fit_transform(X_entren)
X_validacion = escalador.transform(X_validacion)
X_prueba = escalador.transform(X_prueba)

if ESCALAR_OBJETIVO:
    escalador_y = StandardScaler()
    y_entren_modelo = escalador_y.fit_transform(y_entren.reshape(-1, 1)).ravel()
    y_validacion_modelo = escalador_y.transform(y_validacion.reshape(-1, 1)).ravel()
else:
    y_entren_modelo = y_entren
    y_validacion_modelo = y_validacion

print(f"  Archivo: {ARCHIVO_CSV.name}  |  {len(datos)} filas  |  {len(COLUMNAS_ENTRADA)} columnas de entrada")
print(f"  Entrenamiento: {len(X_entren)}  |  Validación: {len(X_validacion)}  |  Prueba: {len(X_prueba)}")
print()

X_entren_tensor = torch.tensor(X_entren, dtype=torch.float32)
y_entren_tensor = torch.tensor(y_entren_modelo, dtype=torch.float32).unsqueeze(1)
X_validacion_tensor = torch.tensor(X_validacion, dtype=torch.float32)
y_validacion_tensor = torch.tensor(y_validacion_modelo, dtype=torch.float32).unsqueeze(1)
X_prueba_tensor = torch.tensor(X_prueba, dtype=torch.float32)

num_entradas = X_entren.shape[1]


# ==============================================================================
# SECCIÓN 3: ARQUITECTURA (la misma red: entrada -> 16 -> 8 -> 1)
# ==============================================================================

class RedNeuronal(nn.Module):
    def __init__(self, num_entradas):
        super(RedNeuronal, self).__init__()
        self.capa1 = nn.Sequential(nn.Linear(num_entradas, 16), nn.ReLU())
        self.capa2 = nn.Sequential(nn.Linear(16, 8), nn.ReLU())
        self.salida = nn.Linear(8, 1)

    def forward(self, x):
        x = self.capa1(x)
        x = self.capa2(x)
        x = self.salida(x)
        return x


# ==============================================================================
# SECCIÓN 4: FUNCIÓN QUE ENTRENA UNA CONFIGURACIÓN UNA VEZ
# ==============================================================================

def entrenar_una_vez(epocas, lr, lote, semilla):
    """
    Entrena la red desde cero y devuelve:
      (pérdida de entrenamiento por época, pérdida de validación por época, MAE de prueba, R² de prueba)
    El mini-lote se arma mezclando los índices en cada época (equivale a DataLoader con shuffle=True).
    """
    torch.manual_seed(semilla)
    red = RedNeuronal(num_entradas)
    perdida_fn = nn.MSELoss()
    optimizador = torch.optim.Adam(red.parameters(), lr=lr)

    n = len(X_entren_tensor)
    historial_entren = []
    historial_valid = []

    for epoca in range(epocas):
        red.train()
        orden = torch.randperm(n)
        perdida_total = 0.0

        for inicio in range(0, n, lote):
            indices = orden[inicio:inicio + lote]
            prediccion = red(X_entren_tensor[indices])
            perdida = perdida_fn(prediccion, y_entren_tensor[indices])
            optimizador.zero_grad()
            perdida.backward()
            optimizador.step()
            perdida_total += perdida.item() * len(indices)

        historial_entren.append(perdida_total / n)

        red.eval()
        with torch.no_grad():
            historial_valid.append(perdida_fn(red(X_validacion_tensor), y_validacion_tensor).item())

    # Evaluación final en datos de prueba, en puntos de calidad reales
    red.eval()
    with torch.no_grad():
        salida = red(X_prueba_tensor).numpy()
    predicciones = escalador_y.inverse_transform(salida).ravel() if ESCALAR_OBJETIVO else salida.ravel()

    if np.isfinite(predicciones).all():
        mae = mean_absolute_error(y_prueba, predicciones)
        r2 = r2_score(y_prueba, predicciones)
    else:  # la red divergió (por ejemplo, con una tasa de aprendizaje demasiado alta)
        mae, r2 = float("nan"), float("nan")

    return np.array(historial_entren), np.array(historial_valid), mae, r2


# ==============================================================================
# SECCIÓN 5: CORRER TODOS LOS EXPERIMENTOS
# ==============================================================================

print("=" * 60)
print("  CORRIENDO EXPERIMENTOS...")
print("=" * 60)

# Cada combinación (épocas, lr, lote) se entrena una sola vez, aunque aparezca en varias familias
resultados = {}
inicio_total = time.time()

for familia, opciones in EXPERIMENTOS.items():
    for etiqueta, cambios in opciones:
        config = {**BASE, **cambios}
        clave = (config["epocas"], config["lr"], config["lote"])
        if clave in resultados:
            continue

        corridas = []
        for k in range(N_SEMILLAS):
            print(f"  [{familia}] {etiqueta:<13} épocas={config['epocas']:<4} lr={config['lr']:<7} "
                  f"lote={config['lote']:<4} semilla {k + 1}/{N_SEMILLAS}", flush=True)
            corridas.append(entrenar_una_vez(config["epocas"], config["lr"], config["lote"], semilla=100 + k))

        resultados[clave] = {
            "train": np.array([c[0] for c in corridas]),   # forma: (semillas, épocas)
            "valid": np.array([c[1] for c in corridas]),
            "mae": np.array([c[2] for c in corridas]),
            "r2": np.array([c[3] for c in corridas]),
        }

print(f"\n  Listo en {time.time() - inicio_total:.0f} segundos.")
print()


# ==============================================================================
# SECCIÓN 6: TABLA DE RESULTADOS
# ==============================================================================

filas = []
listadas = set()
for familia, opciones in EXPERIMENTOS.items():
    for etiqueta, cambios in opciones:
        config = {**BASE, **cambios}
        clave = (config["epocas"], config["lr"], config["lote"])
        if clave in listadas:
            continue
        listadas.add(clave)

        r = resultados[clave]
        valid_media = np.nanmean(r["valid"], axis=0)
        entren_media = np.nanmean(r["train"], axis=0)
        finales_valid = r["valid"][:, -1]
        finales_entren = r["train"][:, -1]

        filas.append({
            "grupo": "Base" if not cambios else familia,
            "configuracion": etiqueta,
            "epocas": config["epocas"],
            "tasa_aprendizaje": config["lr"],
            "lote": config["lote"],
            "mejor_epoca_validacion": int(np.nanargmin(valid_media)) + 1,
            "mejor_mse_validacion": float(np.nanmin(valid_media)),
            "mse_validacion_final": float(np.nanmean(finales_valid)),
            "mse_entrenamiento_final": float(np.nanmean(finales_entren)),
            "brecha_final": float(np.nanmean(finales_valid - finales_entren)),
            "mae_prueba": float(np.nanmean(r["mae"])),
            "mae_prueba_desv": float(np.nanstd(r["mae"])),
            "r2_prueba": float(np.nanmean(r["r2"])),
            "r2_prueba_desv": float(np.nanstd(r["r2"])),
        })

tabla = pd.DataFrame(filas)
tabla.to_csv(ARCHIVO_RESULTADOS, index=False, encoding="utf-8-sig")

print("=" * 60)
print(f"  RESULTADOS (promedio de {N_SEMILLAS} semillas)")
print("=" * 60)
resumen = pd.DataFrame({
    "Grupo": tabla["grupo"],
    "Config": tabla["configuracion"],
    "MejorÉpoca": tabla["mejor_epoca_validacion"],
    "ValidMSE": tabla["mse_validacion_final"].map("{:.3f}".format),
    "TrainMSE": tabla["mse_entrenamiento_final"].map("{:.3f}".format),
    "Brecha": tabla["brecha_final"].map("{:+.3f}".format),
    "MAE prueba": [f"{m:.3f} ± {d:.3f}" for m, d in zip(tabla["mae_prueba"], tabla["mae_prueba_desv"])],
    "R² prueba": [f"{m:.3f} ± {d:.3f}" for m, d in zip(tabla["r2_prueba"], tabla["r2_prueba_desv"])],
})
print(resumen.to_string(index=False))
print()
print("  MejorÉpoca = época donde la validación (promedio) fue más baja.")
print("  Brecha = validación - entrenamiento al final (más grande = más sobreajuste).")
print("  Si dos configuraciones difieren menos que su desviación (±), no hay diferencia real.")

if tabla["mae_prueba"].notna().any():
    mejor = tabla.loc[tabla["mae_prueba"].idxmin()]
    print(f"\n  Menor MAE de prueba: {mejor['configuracion']} ({mejor['grupo']}) con {mejor['mae_prueba']:.3f}")
print(f"  Tabla guardada en: {ARCHIVO_RESULTADOS}")
print()


# ==============================================================================
# SECCIÓN 7: GRÁFICA COMPARATIVA
# ==============================================================================

TOPE_Y = 1.6  # el eje se recorta aquí para que una curva que explota no aplaste las demás

todas = np.concatenate([
    np.nanmean(r[k], axis=0) for r in resultados.values() for k in ("train", "valid")
])
piso_y = max(0.0, float(np.nanmin(todas)) * 0.9)

familias = list(EXPERIMENTOS.keys())
fig, ejes = plt.subplots(2, len(familias), figsize=(5 * len(familias), 8), squeeze=False, sharey=True)
fig.suptitle(f"Efecto de los hiperparámetros (promedio de {N_SEMILLAS} semillas)", fontsize=14)

for col, familia in enumerate(familias):
    for fila, (clave_serie, titulo) in enumerate([("valid", "Validación"), ("train", "Entrenamiento")]):
        eje = ejes[fila][col]
        for etiqueta, cambios in EXPERIMENTOS[familia]:
            config = {**BASE, **cambios}
            r = resultados[(config["epocas"], config["lr"], config["lote"])]
            media = np.nanmean(r[clave_serie], axis=0)
            desv = np.nanstd(r[clave_serie], axis=0)
            x = np.arange(1, len(media) + 1)
            es_base = not cambios
            linea, = eje.plot(x, media, label=etiqueta, linewidth=2.5 if es_base else 1.6,
                              color="black" if es_base else None)
            eje.fill_between(x, media - desv, media + desv, color=linea.get_color(), alpha=0.12)
            if clave_serie == "valid" and np.isfinite(media).any():
                mejor_ep = int(np.nanargmin(media))
                eje.scatter([mejor_ep + 1], [media[mejor_ep]], color=linea.get_color(), s=22, zorder=5)
        eje.axhline(1.0, color="gray", linestyle=":", linewidth=1)
        eje.set_ylim(piso_y, TOPE_Y)
        eje.grid(True, linestyle=":", alpha=0.6)
        eje.set_title(f"{familia} — {titulo}")
        if fila == 1:
            eje.set_xlabel("Época")
        if col == 0:
            eje.set_ylabel("MSE (objetivo escalado)" if ESCALAR_OBJETIVO else "MSE")
        eje.legend(title="Valor", fontsize=8)

fig.text(0.5, 0.005,
         "Línea punteada gris = predecir siempre el promedio (1.0). Puntos = mejor época de validación. "
         f"Eje recortado en {TOPE_Y}.",
         ha="center", fontsize=9)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig(f"{ARCHIVO_COMPARACION}.png", dpi=200)
fig.savefig(f"{ARCHIVO_COMPARACION}.pdf")
plt.close(fig)

print("=" * 60)
print(f"  Gráfica guardada en: {ARCHIVO_COMPARACION}.png / .pdf")
print("=" * 60)
