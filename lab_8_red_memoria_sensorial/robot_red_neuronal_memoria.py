import argparse
import os
import random
import time
from copy import deepcopy

import torch
import torch.nn as nn


# ==========================================================
# LABORATORIO 8: RED NEURONAL EVOLUTIVA CON MEMORIA SENSORIAL
# ==========================================================
# Base: robot_red_neuronal_evolutiva.py (laboratorio 7).
#
# QUE SE CONSERVA (igual que en el lab 7):
#   - el mundo: cuadricula 8x8, INICIO, META, obstaculos, sensores locales
#   - el algoritmo genetico: elitismo, cruce en un punto, mutacion gaussiana
#   - el formato del fitness (se le agrega un termino)
#
# QUE CAMBIA (marcado con "LAB 8" en los comentarios):
#   1. La entrada de la red pasa de 4 a 12 valores: s_t, s_{t-1}, a_{t-1}.
#   2. La capa oculta pasa de 8 a 16 neuronas.
#   3. recorrer() recuerda la observacion y la accion del paso anterior.
#   4. El fitness resta 10*b, donde b = bucles sensoriomotores.
#   5. Hay mapas nuevos (pasillos estrechos y bucles locales) y se puede
#      elegir entre red basica (4 entradas) y red con memoria (12 entradas)
#      para poder compararlas en igualdad de condiciones.
#   6. Se fija tambien la semilla de PyTorch (en el lab 7 solo se fijaba la de
#      random, asi que los pesos iniciales no eran reproducibles).
# ==========================================================

COMANDOS = ("U", "D", "L", "R")
ORDEN_SENSORES = ("U", "D", "L", "R")

TAMANO = 8
INICIO = (0, 0)
META = (7, 7)

# LAB 8: varios mapas con el mismo TAMANO, INICIO y META.
# "sencillo" es el mapa original del lab 7 y "mapa2" el mapa modificado.
MAPAS = {
    "sencillo": {
        (0, 3), (1, 3), (2, 0), (2, 2), (2, 3),
        (4, 5), (5, 2), (5, 3), (5, 4),
        (6, 6),
    },
    "mapa2": {
        (0, 5), (1, 1), (1, 5), (2, 1), (2, 4), (2, 5),
        (3, 1), (4, 3), (4, 4), (5, 6), (6, 1), (6, 2),
    },
    # Pasillos de ancho 1: el robot debe avanzar en zig-zag.
    "pasillos": {
        (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
        (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
        (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7),
    },
    # Bolsillos en forma de "C": un robot reactivo entra y se queda oscilando.
    "bucles": {
        (2, 2), (2, 3), (2, 4), (3, 2), (4, 2), (4, 3), (4, 4),
        (5, 6), (6, 5), (5, 5),
    },
}
OBSTACULOS = MAPAS["mapa2"]

PASOS_MAXIMOS = 30
TAMANO_POBLACION = 40
ELITE = 6
PADRES = 16

# LAB 8: tamanos de entrada y capa oculta.
ENTRADA_BASICA = 4      # s_t
ENTRADA_MEMORIA = 12    # s_t (4) + s_{t-1} (4) + a_{t-1} one-hot (4)
HIDDEN = 16

EMOJI_ROBOT = "\U0001f916"
EMOJI_META = "\U0001f7e9"
EMOJI_META_ALCANZADA = "\U0001f389"
EMOJI_OBSTACULO = "\u2b1b"
EMOJI_INICIO = "\U0001f535"
EMOJI_LIBRE = "\u2b1c"


# ==========================================================
# 2. MODELO NEURONAL (PYTORCH)
# ==========================================================
# LAB 8: la red ahora puede tener 4 entradas (basica, como en el lab 7) o 12
# (con memoria sensorial). Todo lo demas (ReLU, 4 logits, argmax) es igual.
class RedPercepcion(nn.Module):
    """Red con entrada de 4 (basica) o 12 (memoria) valores y 4 acciones."""

    def __init__(self, hidden_size=HIDDEN, usar_memoria=True):
        super().__init__()
        self.usar_memoria = usar_memoria
        self.hidden_size = hidden_size
        entrada = ENTRADA_MEMORIA if usar_memoria else ENTRADA_BASICA
        self.hidden = nn.Linear(entrada, hidden_size)
        self.out = nn.Linear(hidden_size, 4)

    def forward(self, x):
        return self.out(torch.relu(self.hidden(x)))

    def vector(self):
        return torch.cat([p.detach().reshape(-1) for p in self.parameters()])

    def cargar_vector(self, vector):
        with torch.no_grad():
            pos = 0
            for parametro in self.parameters():
                largo = parametro.numel()
                parametro.copy_(vector[pos:pos + largo].reshape_as(parametro))
                pos += largo


def crear_red_aleatoria(rng, usar_memoria=True):
    red = RedPercepcion(usar_memoria=usar_memoria)
    for parametro in red.parameters():
        parametro.data.normal_(mean=0.0, std=0.5)
    return red


def copiar_red(red):
    copia = RedPercepcion(hidden_size=red.hidden_size, usar_memoria=red.usar_memoria)
    copia.load_state_dict(deepcopy(red.state_dict()))
    return copia


def guardar_red(red, ruta, mapa_entrenamiento=None):
    directorio = os.path.dirname(ruta)
    if directorio:
        os.makedirs(directorio, exist_ok=True)

    estado = {
        "state_dict": red.state_dict(),
        "hidden_size": red.hidden_size,
        "usar_memoria": red.usar_memoria,  # LAB 8: tipo de red
        "comandos": COMANDOS,
        "orden_sensores": ORDEN_SENSORES,
        "tamano": TAMANO,
        "inicio": INICIO,
        "meta": META,
        "mapa_entrenamiento": mapa_entrenamiento,
    }
    torch.save(estado, ruta)
    return ruta


def cargar_red(ruta):
    checkpoint = torch.load(ruta, map_location="cpu")
    if isinstance(checkpoint, nn.Module):
        return checkpoint

    red = RedPercepcion(
        hidden_size=checkpoint.get("hidden_size", HIDDEN),
        usar_memoria=checkpoint.get("usar_memoria", True),
    )
    red.load_state_dict(checkpoint["state_dict"])
    red.eval()
    return red


# ==========================================================
# 3. ENTORNO Y SENSADO
# ==========================================================
def limpiar_pantalla():
    os.system("cls" if os.name == "nt" else "clear")


def destino_de(posicion, comando):
    fila, columna = posicion
    cambios = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
    cambio_fila, cambio_columna = cambios[comando]
    return fila + cambio_fila, columna + cambio_columna


def esta_bloqueada(posicion, obstaculos=None):
    fila, columna = posicion
    obstaculos = OBSTACULOS if obstaculos is None else set(obstaculos)
    return not (0 <= fila < TAMANO and 0 <= columna < TAMANO) or posicion in obstaculos


def observar(posicion, obstaculos=None):
    """Codifica los cuatro sensores en un entero de 4 bits."""
    patron = 0
    for comando in ORDEN_SENSORES:
        patron <<= 1
        patron |= int(esta_bloqueada(destino_de(posicion, comando), obstaculos=obstaculos))
    return patron


def bits_de(observacion):
    """Convierte el entero de 4 bits en la lista [U, D, L, R]."""
    return [(observacion >> (3 - i)) & 1 for i in range(4)]


def one_hot(accion):
    """LAB 8: codifica la accion previa en 4 bits. Sin accion previa -> ceros."""
    if accion is None:
        return [0, 0, 0, 0]
    return [1 if accion == c else 0 for c in COMANDOS]


def construir_entrada(red, obs_actual, obs_previa, accion_previa):
    """LAB 8: arma el vector de entrada de la red.

    Basica:   [s_t]                        -> 4 valores
    Memoria:  [s_t, s_{t-1}, a_{t-1}]      -> 12 valores
    En t = 0 no hay pasado: s_{t-1} y a_{t-1} son ceros.
    """
    entrada = bits_de(obs_actual)
    if red.usar_memoria:
        previa = bits_de(obs_previa) if obs_previa is not None else [0, 0, 0, 0]
        entrada = entrada + previa + one_hot(accion_previa)
    return torch.tensor(entrada, dtype=torch.float32).unsqueeze(0)


def decidir(red, obs_actual, obs_previa=None, accion_previa=None):
    """Pasa la entrada (con o sin memoria) por la red y devuelve la accion."""
    entrada = construir_entrada(red, obs_actual, obs_previa, accion_previa)
    with torch.no_grad():
        logits = red(entrada)[0]
    indice = int(torch.argmax(logits).item())
    return COMANDOS[indice]


def mover(posicion, comando, obstaculos=None, tamano=None):
    destino = destino_de(posicion, comando)
    tamano = TAMANO if tamano is None else tamano
    obstaculos = OBSTACULOS if obstaculos is None else set(obstaculos)
    if not (0 <= destino[0] < tamano and 0 <= destino[1] < tamano):
        return posicion, "borde"
    if destino in obstaculos:
        return posicion, "obstaculo"
    return destino, "avance"


def recorrer(red, obstaculos=None, inicio=None, meta=None, pasos_maximos=None, tamano=None):
    """Ejecuta una trayectoria completa. Devuelve un diccionario de resultados."""
    obstaculos = OBSTACULOS if obstaculos is None else set(obstaculos)
    inicio = INICIO if inicio is None else inicio
    meta = META if meta is None else meta
    pasos_maximos = PASOS_MAXIMOS if pasos_maximos is None else pasos_maximos
    tamano = TAMANO if tamano is None else tamano

    posicion = inicio
    trayectoria = [posicion]
    observaciones = []
    acciones = []
    choques = 0
    pasos_utiles = 0

    # LAB 8: memoria de corto plazo (s_{t-1} y a_{t-1}).
    obs_previa = None
    accion_previa = None
    # LAB 8: pares (posicion, accion) ya vistos para detectar bucles.
    vistos = set()
    bucles = 0

    for _ in range(pasos_maximos):
        observacion = observar(posicion, obstaculos=obstaculos)
        comando = decidir(red, observacion, obs_previa, accion_previa)

        # LAB 8: si el robot repite exactamente la misma situacion (misma celda
        # y misma accion) esta recorriendo un bucle sensoriomotor.
        if (posicion, comando) in vistos:
            bucles += 1
        vistos.add((posicion, comando))

        nueva_posicion, resultado = mover(posicion, comando, obstaculos=obstaculos, tamano=tamano)

        observaciones.append(observacion)
        acciones.append(comando)

        if resultado in ("borde", "obstaculo"):
            choques += 1
        if resultado == "avance":
            pasos_utiles += 1

        # LAB 8: actualizar la memoria DESPUES de actuar.
        obs_previa = observacion
        accion_previa = comando

        posicion = nueva_posicion
        trayectoria.append(posicion)
        if posicion == meta:
            break

    return {
        "trayectoria": trayectoria,
        "choques": choques,
        "pasos_utiles": pasos_utiles,
        "observaciones": observaciones,
        "acciones": acciones,
        "bucles": bucles,
    }


# ==========================================================
# 4. FITNESS Y EVOLUCION
# ==========================================================
# F = 500 - 35d + 4p - 30c - 3r + 2000m - 10b
#   d distancia Manhattan final, p pasos utiles, c choques,
#   r posiciones repetidas, m meta alcanzada, b bucles (LAB 8).
def puntaje_de(res, meta=None):
    meta = META if meta is None else meta
    trayectoria = res["trayectoria"]
    posicion = trayectoria[-1]
    distancia = abs(meta[0] - posicion[0]) + abs(meta[1] - posicion[1])
    repetidas = len(trayectoria) - len(set(trayectoria))

    puntaje = 500
    puntaje -= distancia * 35
    puntaje += res["pasos_utiles"] * 4
    puntaje -= res["choques"] * 30
    puntaje -= repetidas * 3
    puntaje -= res["bucles"] * 10  # LAB 8: penalizacion por bucles
    if posicion == meta:
        puntaje += 2000
    return puntaje


def evaluar(red, obstaculos=None, inicio=None, meta=None, pasos_maximos=None, tamano=None):
    res = recorrer(red, obstaculos=obstaculos, inicio=inicio, meta=meta, pasos_maximos=pasos_maximos, tamano=tamano)
    return puntaje_de(res, meta=meta)


def metricas(red, obstaculos=None, inicio=None, meta=None, pasos_maximos=None, tamano=None):
    """LAB 8: metricas para la tabla de resultados."""
    meta_ = META if meta is None else meta
    res = recorrer(red, obstaculos=obstaculos, inicio=inicio, meta=meta, pasos_maximos=pasos_maximos, tamano=tamano)
    final = res["trayectoria"][-1]
    return {
        "exito": final == meta_,
        "choques": res["choques"],
        "pasos": len(res["trayectoria"]) - 1,
        "ciclos": res["bucles"],
        "distancia_final": abs(meta_[0] - final[0]) + abs(meta_[1] - final[1]),
        "puntaje": puntaje_de(res, meta=meta),
    }


def mutar_red(red, tasa_mutacion, sigma, rng):
    red_mutada = copiar_red(red)
    with torch.no_grad():
        for parametro in red_mutada.parameters():
            ruido = torch.randn_like(parametro) * sigma
            mascara = torch.rand_like(parametro) < tasa_mutacion
            parametro.add_(ruido * mascara.to(parametro.dtype))
    return red_mutada


def cruzar_redes(padre_1, padre_2, rng):
    v1 = padre_1.vector()
    v2 = padre_2.vector()
    punto = rng.randint(1, len(v1) - 1)
    hijo = v1.clone()
    hijo[punto:] = v2[punto:]

    red_hijo = RedPercepcion(hidden_size=padre_1.hidden_size, usar_memoria=padre_1.usar_memoria)
    red_hijo.cargar_vector(hijo)
    return red_hijo


def evolucionar(tasa_mutacion=0.15, usar_cruce=True, semilla=None, maximo_generaciones=40,
                usar_memoria=True, obstaculos=None):
    """Ciclo evolutivo completo. Devuelve el mejor individuo."""
    obstaculos = OBSTACULOS if obstaculos is None else set(obstaculos)
    if semilla is None:
        semilla = random.randint(1, 1000000)
    rng = random.Random(semilla)
    torch.manual_seed(semilla)  # LAB 8: pesos iniciales y mutaciones reproducibles

    poblacion = [crear_red_aleatoria(rng, usar_memoria=usar_memoria) for _ in range(TAMANO_POBLACION)]
    historial = []

    def ordenar(poblacion):
        # Se evalua una sola vez por individuo y generacion.
        puntuados = [(evaluar(r, obstaculos=obstaculos), r) for r in poblacion]
        puntuados.sort(key=lambda par: par[0], reverse=True)
        return [r for _, r in puntuados], [p for p, _ in puntuados]

    for generacion in range(maximo_generaciones + 1):
        poblacion, puntajes = ordenar(poblacion)
        mejor = poblacion[0]
        res = recorrer(mejor, obstaculos=obstaculos)
        historial.append((generacion, puntajes[0]))

        if res["trayectoria"][-1] == META:
            return {
                "metodo": "red neuronal + evolutivo" + (" (memoria)" if usar_memoria else " (basica)"),
                "red": mejor, "generacion": generacion, "puntaje": puntajes[0],
                "choques": res["choques"], "pasos_utiles": res["pasos_utiles"],
                "bucles": res["bucles"], "llego": True,
                "historial": historial, "poblacion": poblacion,
            }

        nueva_poblacion = [copiar_red(red) for red in poblacion[:ELITE]]
        while len(nueva_poblacion) < TAMANO_POBLACION:
            madre = rng.choice(poblacion[:PADRES])
            if usar_cruce and rng.random() < 0.75:
                padre = rng.choice(poblacion[:PADRES])
                hijo = cruzar_redes(madre, padre, rng)
            else:
                hijo = copiar_red(madre)
            nueva_poblacion.append(mutar_red(hijo, tasa_mutacion, sigma=0.45, rng=rng))
        poblacion = nueva_poblacion

    poblacion, puntajes = ordenar(poblacion)
    mejor = poblacion[0]
    res = recorrer(mejor, obstaculos=obstaculos)
    return {
        "metodo": "red neuronal + evolutivo" + (" (memoria)" if usar_memoria else " (basica)"),
        "red": mejor, "generacion": maximo_generaciones, "puntaje": puntajes[0],
        "choques": res["choques"], "pasos_utiles": res["pasos_utiles"],
        "bucles": res["bucles"], "llego": res["trayectoria"][-1] == META,
        "historial": historial, "poblacion": poblacion,
    }


# ==========================================================
# 5. VISUALIZACION
# ==========================================================
def dibujar(red, pausa=0.1, titulo="MEJOR INDIVIDUO", obstaculos=None, inicio=None, meta=None, pasos_maximos=None, tamano=None):
    obstaculos = OBSTACULOS if obstaculos is None else set(obstaculos)
    inicio = INICIO if inicio is None else inicio
    meta = META if meta is None else meta
    tamano = TAMANO if tamano is None else tamano

    res = recorrer(red, obstaculos=obstaculos, inicio=inicio, meta=meta, pasos_maximos=pasos_maximos, tamano=tamano)
    trayectoria, observaciones, acciones = res["trayectoria"], res["observaciones"], res["acciones"]
    puntaje = puntaje_de(res, meta=meta)
    tipo = "MEMORIA" if red.usar_memoria else "BASICA"

    for paso, posicion in enumerate(trayectoria[1:], start=1):
        limpiar_pantalla()
        print(f"RED {tipo} | {titulo} | Puntaje: {puntaje}")
        print(f"{EMOJI_ROBOT} = robot | {EMOJI_META_ALCANZADA} = meta alcanzada | {EMOJI_META} = meta | {EMOJI_OBSTACULO} = obstaculo")
        print(f"{EMOJI_INICIO} = inicio | {EMOJI_LIBRE} = celda libre\n")

        for fila in range(tamano):
            linea = ""
            for columna in range(tamano):
                celda = (fila, columna)
                if celda == posicion:
                    linea += EMOJI_ROBOT
                elif celda == inicio:
                    linea += EMOJI_INICIO
                elif celda == meta:
                    linea += EMOJI_META_ALCANZADA if posicion == meta else EMOJI_META
                elif celda in obstaculos:
                    linea += EMOJI_OBSTACULO
                else:
                    linea += EMOJI_LIBRE
            print(linea)

        print(f"Paso {paso}/{len(trayectoria)-1} | Observacion: {observaciones[paso - 1]:04b} | Accion: {acciones[paso - 1]}")
        if pausa:
            time.sleep(pausa)


def main():
    parser = argparse.ArgumentParser(description="Red neuronal con memoria sensorial evolucionada por algoritmo genetico.")
    parser.add_argument("--semilla", type=int, default=None)
    parser.add_argument("--generaciones", type=int, default=40)
    parser.add_argument("--tasa-mutacion", type=float, default=0.15)
    parser.add_argument("--sin-pausa", action="store_true")
    parser.add_argument("--sin-animacion", action="store_true", help="No dibuja la animacion, solo imprime el resultado.")
    parser.add_argument("--guardar-mejor", type=str, default=None)
    parser.add_argument("--cargar-red", type=str, default=None)
    # LAB 8: nuevas opciones
    parser.add_argument("--basica", action="store_true", help="Usa la red basica de 4 entradas (sin memoria).")
    parser.add_argument("--mapa", type=str, default="mapa2", choices=sorted(MAPAS), help="Mapa de entrenamiento / evaluacion.")
    args = parser.parse_args()

    pausa = 0 if args.sin_pausa else 0.08
    obstaculos = MAPAS[args.mapa]

    if args.cargar_red:
        red = cargar_red(args.cargar_red)
        res = recorrer(red, obstaculos=obstaculos)
        resultado = {
            "metodo": "red neuronal cargada" + (" (memoria)" if red.usar_memoria else " (basica)"),
            "red": red, "generacion": "cargado",
            "puntaje": puntaje_de(res), "choques": res["choques"],
            "pasos_utiles": res["pasos_utiles"], "bucles": res["bucles"],
            "llego": res["trayectoria"][-1] == META,
        }
    else:
        resultado = evolucionar(
            tasa_mutacion=args.tasa_mutacion, usar_cruce=True, semilla=args.semilla,
            maximo_generaciones=args.generaciones, usar_memoria=not args.basica,
            obstaculos=obstaculos,
        )
        if args.guardar_mejor:
            guardar_red(resultado["red"], args.guardar_mejor, mapa_entrenamiento=args.mapa)
            print(f"\nMejor individuo guardado en: {args.guardar_mejor}")

    if not args.sin_animacion:
        print("\nANIMACION DEL MEJOR INDIVIDUO")
        dibujar(resultado["red"], pausa=pausa, titulo=f"MAPA {args.mapa}", obstaculos=obstaculos)

    m = metricas(resultado["red"], obstaculos=obstaculos)
    print("\nRESULTADO FINAL")
    print(f"Metodo: {resultado['metodo']}")
    print(f"Mapa: {args.mapa}")
    print(f"Generacion: {resultado['generacion']}")
    print(f"Puntaje: {resultado['puntaje']}")
    print(f"Choques: {resultado['choques']}")
    print(f"Pasos utiles: {resultado['pasos_utiles']}")
    print(f"Ciclos (bucles): {resultado['bucles']}")
    print(f"Distancia final: {m['distancia_final']}")
    print(f"Llegó a la meta: {'si' if resultado['llego'] else 'no'}")


if __name__ == "__main__":
    main()
