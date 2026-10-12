"""
================================================================================
  RED NEURONAL CON PYTORCH - PREDICCIÓN DE CALIDAD DE VINO TINTO
================================================================================

Adaptación de red_neuronal.py (predicción de demanda de comida) a Red Wine Quality.
Es REGRESIÓN, igual que la plantilla original: MSELoss, MAE y R².
Mantiene la misma estructura y los mismos hiperparámetros base
(100 épocas, lr=0.001, lote 64, Adam, arquitectura entrada -> 16 -> 8 -> 1).

Guarda historial_entrenamiento.csv y curva_aprendizaje (.png y .pdf).

QUÉ CAMBIA RESPECTO A LA PLANTILLA ORIGINAL:
  1. Busca solo el CSV: cualquier .csv con "wine" en el nombre dentro de esta carpeta.
  2. Detecta el separador (coma o punto y coma) y usa TODAS las columnas como
     entrada, menos la objetivo ("quality") y una posible columna "Id".
  3. Escala también el objetivo (ESCALAR_OBJETIVO). Así la pérdida se interpreta fácil:
       pérdida = 1.0  ->  la red solo predice el promedio (no aprendió nada)
       pérdida < 1.0  ->  aprendió algo; cuanto más baja, mejor
     Las métricas finales (MAE, R²) se calculan igualmente en puntos de calidad.
  4. Compara contra una regresión lineal, para ver si la red aporta algo.

REQUISITOS:
  pip install torch pandas scikit-learn matplotlib
================================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# "Agg" guarda las gráficas en archivo sin abrir ventanas (evita problemas en terminal).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==============================================================================
# SECCIÓN 1: CONFIGURACIÓN
# ==============================================================================

CARPETA_SCRIPT = Path(__file__).resolve().parent

# None = busca automáticamente un .csv con "wine" en el nombre.
# Si tu archivo se llama distinto, escribe: ARCHIVO_CSV = CARPETA_SCRIPT / "mi_archivo.csv"
ARCHIVO_CSV = None
ARCHIVO_HISTORIAL = CARPETA_SCRIPT / "historial_entrenamiento.csv"
ARCHIVO_CURVA = CARPETA_SCRIPT / "curva_aprendizaje"  # se guarda .png y .pdf

COLUMNA_OBJETIVO = "quality"       # Lo que queremos predecir (calificación del vino)
COLUMNAS_IGNORAR = ["Id", "id"]    # Columnas que no sirven como entrada (si existen)

# Hiperparámetros (son los que se analizan en las diapositivas del profe)
EPOCAS = 100
TASA_APRENDIZAJE = 0.001
TAMANO_LOTE = 64
PORCENTAJE_PRUEBA = 0.2
PORCENTAJE_VALIDACION = 0.2
SEMILLA = 42                       # Hace el experimento reproducible

# ¿POR QUÉ escalar el objetivo?
#   La red arranca prediciendo cerca de 0, pero la calidad ronda 5-6. Con lr=0.001 gasta
#   muchas épocas solo en "llegar" a ese valor, y la curva queda plana al inicio.
#   Con el objetivo escalado (media 0, desviación 1) eso desaparece.
ESCALAR_OBJETIVO = True

np.random.seed(SEMILLA)
torch.manual_seed(SEMILLA)

if ARCHIVO_CSV is None:
    candidatos = sorted(p for p in CARPETA_SCRIPT.glob("*.csv") if "wine" in p.name.lower())
    if not candidatos:
        raise FileNotFoundError(
            "No encontré ningún .csv con 'wine' en el nombre en esta carpeta. "
            "Ponlo junto al script o escribe su nombre en ARCHIVO_CSV."
        )
    ARCHIVO_CSV = candidatos[0]


# ==============================================================================
# SECCIÓN 2: CARGAR Y PREPARAR LOS DATOS
# ==============================================================================

print("=" * 60)
print("  CARGANDO DATOS...")
print("=" * 60)

# Algunas versiones del archivo usan ";" como separador y otras ","
with open(ARCHIVO_CSV, encoding="utf-8-sig") as f:
    primera_linea = f.readline()
separador = ";" if primera_linea.count(";") > primera_linea.count(",") else ","

datos = pd.read_csv(ARCHIVO_CSV, sep=separador)
datos.columns = datos.columns.str.strip()

if COLUMNA_OBJETIVO not in datos.columns:
    raise ValueError(
        f"No existe la columna '{COLUMNA_OBJETIVO}'. Columnas disponibles: {list(datos.columns)}"
    )

datos = datos.drop(columns=[c for c in COLUMNAS_IGNORAR if c in datos.columns])
filas_antes = len(datos)
datos = datos.dropna()

COLUMNAS_ENTRADA = [c for c in datos.columns if c != COLUMNA_OBJETIVO]

print(f"  Archivo:                {ARCHIVO_CSV.name}")
print(f"  Filas totales:          {len(datos)} (se descartaron {filas_antes - len(datos)} con vacíos)")
print(f"  Columnas de entrada:    {len(COLUMNAS_ENTRADA)}")
print(f"    {COLUMNAS_ENTRADA}")
print(f"  Columna objetivo:       {COLUMNA_OBJETIVO}")
print(f"  Calidad promedio:       {datos[COLUMNA_OBJETIVO].mean():.2f}")
print(f"  Calidad mín / máx:      {datos[COLUMNA_OBJETIVO].min()} / {datos[COLUMNA_OBJETIVO].max()}")
print()

X = datos[COLUMNAS_ENTRADA].to_numpy(dtype=float)
y = datos[COLUMNA_OBJETIVO].to_numpy(dtype=float)

X_ent_val, X_prueba, y_ent_val, y_prueba = train_test_split(
    X, y, test_size=PORCENTAJE_PRUEBA, random_state=SEMILLA
)
X_entren, X_validacion, y_entren, y_validacion = train_test_split(
    X_ent_val, y_ent_val, test_size=PORCENTAJE_VALIDACION, random_state=SEMILLA
)

# fit_transform solo en entrenamiento; transform en validación y prueba (evita data leakage)
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

print(f"  Datos de entrenamiento: {len(X_entren)} filas")
print(f"  Datos de validación:    {len(X_validacion)} filas")
print(f"  Datos de prueba:        {len(X_prueba)} filas")
print()


# ==============================================================================
# SECCIÓN 3: CONVERTIR DATOS A TENSORES DE PYTORCH
# ==============================================================================

X_entren_tensor = torch.tensor(X_entren, dtype=torch.float32)
y_entren_tensor = torch.tensor(y_entren_modelo, dtype=torch.float32).unsqueeze(1)
X_validacion_tensor = torch.tensor(X_validacion, dtype=torch.float32)
y_validacion_tensor = torch.tensor(y_validacion_modelo, dtype=torch.float32).unsqueeze(1)
X_prueba_tensor = torch.tensor(X_prueba, dtype=torch.float32)

cargador_entren = DataLoader(
    TensorDataset(X_entren_tensor, y_entren_tensor),
    batch_size=TAMANO_LOTE,
    shuffle=True
)


# ==============================================================================
# SECCIÓN 4: DEFINIR LA ARQUITECTURA DE LA RED NEURONAL
# ==============================================================================

class RedNeuronal(nn.Module):
    """
    Misma arquitectura de la plantilla: entrada -> 16 -> 8 -> 1.
    La salida no lleva activación porque es regresión (cualquier número real).
    """

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


num_entradas = X_entren.shape[1]
red = RedNeuronal(num_entradas)

print(f"  Arquitectura: {num_entradas} entradas → 16 → 8 → 1 salida")
print(f"  Parámetros totales a aprender: {sum(p.numel() for p in red.parameters())}")
print()


# ==============================================================================
# SECCIÓN 5: FUNCIÓN DE PÉRDIDA Y OPTIMIZADOR
# ==============================================================================

perdida_fn = nn.MSELoss()
optimizador = torch.optim.Adam(red.parameters(), lr=TASA_APRENDIZAJE)


# ==============================================================================
# SECCIÓN 6: BUCLE DE ENTRENAMIENTO
# ==============================================================================

print("=" * 60)
print("  ENTRENANDO LA RED...")
print("=" * 60)

historial_entrenamiento = []

for epoca in range(EPOCAS):
    red.train()
    perdida_total = 0.0

    for X_lote, y_lote in cargador_entren:
        prediccion = red(X_lote)                  # 1. Forward pass
        perdida = perdida_fn(prediccion, y_lote)  # 2. Pérdida
        optimizador.zero_grad()                   # 3. Limpiar gradientes
        perdida.backward()                        # 4. Backward pass
        optimizador.step()                        # 5. Actualizar pesos
        perdida_total += perdida.item() * len(X_lote)

    perdida_entrenamiento = perdida_total / len(X_entren_tensor)

    # Validación: mide generalización; no calcula gradientes ni cambia los pesos
    red.eval()
    with torch.no_grad():
        prediccion_validacion = red(X_validacion_tensor)
        perdida_validacion = perdida_fn(prediccion_validacion, y_validacion_tensor).item()

    historial_entrenamiento.append({
        "epoca": epoca + 1,
        "train_mse": perdida_entrenamiento,
        "validation_mse": perdida_validacion,
        "learning_rate": optimizador.param_groups[0]["lr"]
    })

    if (epoca + 1) % 10 == 0:
        print(
            f"  Época [{epoca + 1:3d}/{EPOCAS}]  |  "
            f"MSE entrenamiento: {perdida_entrenamiento:.4f}  |  "
            f"MSE validación: {perdida_validacion:.4f}"
        )

pd.DataFrame(historial_entrenamiento).to_csv(ARCHIVO_HISTORIAL, index=False)
print(f"  Historial de aprendizaje guardado en: {ARCHIVO_HISTORIAL}")
print()


# ==============================================================================
# SECCIÓN 7: EVALUACIÓN EN DATOS DE PRUEBA
# ==============================================================================

print("=" * 60)
print("  EVALUANDO EL MODELO EN DATOS NO VISTOS...")
print("=" * 60)

red.eval()
with torch.no_grad():
    salida_modelo = red(X_prueba_tensor).numpy()

# Si el objetivo se escaló, se regresa a puntos de calidad reales
if ESCALAR_OBJETIVO:
    predicciones = escalador_y.inverse_transform(salida_modelo).ravel()
else:
    predicciones = salida_modelo.ravel()
reales = y_prueba

mae = mean_absolute_error(reales, predicciones)
r2 = r2_score(reales, predicciones)

# Referencias para saber si la red realmente aporta algo
mae_promedio = np.abs(reales - y_entren.mean()).mean()
regresion_lineal = LinearRegression().fit(X_entren, y_entren)
r2_lineal = r2_score(reales, regresion_lineal.predict(X_prueba))

print(f"  Error Absoluto Medio (MAE): {mae:.3f} puntos de calidad")
print(f"  → La red se equivoca ±{mae:.2f} puntos en promedio")
print(f"  → Referencia: predecir siempre el promedio daría MAE = {mae_promedio:.3f}")
print()
print(f"  R² (coeficiente de determinación): {r2:.3f}")
print(f"  → Referencia: una regresión lineal obtiene R² = {r2_lineal:.3f}")
print("  → (R² = 0 es predecir el promedio; la calidad del vino es subjetiva,")
print("     por eso en este dataset es normal un R² moderado)")
print()


# ==============================================================================
# SECCIÓN 8: EJEMPLOS DE PREDICCIÓN
# ==============================================================================

print("=" * 60)
print("  EJEMPLOS: PREDICCIÓN vs VALOR REAL")
print("=" * 60)
print(f"  {'#':<4} {'Real':>6} {'Predicción':>11} {'Error':>8}")
print(f"  {'-' * 32}")

for i in range(min(30, len(reales))):
    print(f"  {i + 1:<4} {reales[i]:>6.0f} {predicciones[i]:>11.2f} {abs(reales[i] - predicciones[i]):>8.2f}")
print()


# ==============================================================================
# SECCIÓN 9: CURVA DE APRENDIZAJE
# ==============================================================================

hist = pd.DataFrame(historial_entrenamiento)
mejor = hist.loc[hist["validation_mse"].idxmin()]
mejor_epoca = int(mejor["epoca"])

fig, ejes = plt.subplots(
    3, 1, figsize=(10, 8), sharex=True,
    gridspec_kw={"height_ratios": [3, 1.5, 1]}
)
fig.suptitle("Evolución del aprendizaje de la red neuronal", fontsize=14)

ejes[0].plot(hist["epoca"], hist["train_mse"], label="Entrenamiento", linewidth=2)
ejes[0].plot(hist["epoca"], hist["validation_mse"], label="Validación", linewidth=2)
ejes[0].scatter([mejor_epoca], [mejor["validation_mse"]], zorder=5,
                label=f"Mejor validación: época {mejor_epoca}")
ejes[0].axvline(mejor_epoca, linestyle="--", alpha=0.5)
if ESCALAR_OBJETIVO:
    ejes[0].axhline(1.0, color="gray", linestyle=":", label="Referencia: predecir el promedio (1.0)")
ejes[0].set_ylabel("MSE (objetivo escalado)" if ESCALAR_OBJETIVO else "MSE")
ejes[0].set_title("Pérdida por época: menor suele ser mejor")
ejes[0].legend()
ejes[0].grid(True, linestyle=":", alpha=0.6)

brecha = hist["validation_mse"] - hist["train_mse"]
ejes[1].plot(hist["epoca"], brecha, color="purple", linewidth=2)
ejes[1].fill_between(hist["epoca"], brecha, 0, color="red", alpha=0.15)
ejes[1].axhline(0, color="black", linestyle="--", linewidth=1)
ejes[1].set_ylabel("Validación - entrenamiento")
ejes[1].set_title("Brecha de generalización: vigilar si aumenta")
ejes[1].grid(True, linestyle=":", alpha=0.6)

ejes[2].plot(hist["epoca"], hist["learning_rate"], color="darkorange", linewidth=2)
lr_min, lr_max = hist["learning_rate"].min(), hist["learning_rate"].max()
if lr_min == lr_max:
    ejes[2].set_ylim(lr_min * 0.9, lr_max * 1.1)
ejes[2].set_ylabel("Tasa de aprendizaje")
ejes[2].set_title("Tasa de aprendizaje aplicada")
ejes[2].set_xlabel("Época")
ejes[2].grid(True, linestyle=":", alpha=0.6)

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{ARCHIVO_CURVA}.png", dpi=200)
fig.savefig(f"{ARCHIVO_CURVA}.pdf")
plt.close(fig)

print("=" * 60)
print(f"  Curva de aprendizaje guardada en: {ARCHIVO_CURVA}.png / .pdf")
print("=" * 60)