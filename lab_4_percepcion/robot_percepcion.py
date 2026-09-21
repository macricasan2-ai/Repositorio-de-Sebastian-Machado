import argparse
import json
import os
import random
import re
import time

# ==========================================================
# 1. CONFIGURACION DEL PROBLEMA
# ==========================================================
COMANDOS = ("U", "D", "L", "R")
ORDEN_SENSORES = ("U", "D", "L", "R")
TAMANO = 8
INICIO = (0, 0)
META = (7, 7)
OBSTACULOS = {
    (0, 3), (1, 3), (2, 0), (2, 2), (2, 3),
    (4, 5), (5, 2), (5, 3), (5, 4),
    (6, 6),
}

# MODIFICACION 4: segunda distribucion de obstaculos, para poner a
# prueba la MISMA politica ya entrenada en un mapa distinto, sin volver
# a evolucionar. INICIO y META se mantienen iguales en ambos mapas.
OBSTACULOS_MAPA_2 = {
    (0, 5), (1, 1), (1, 5), (2, 1), (2, 4), (2, 5),
    (3, 1), (4, 3), (4, 4), (5, 6), (6, 1), (6, 2),
}

MAPAS = {
    "original": OBSTACULOS,
    "alternativo": OBSTACULOS_MAPA_2,
}


def usar_mapa(nombre_mapa):
    """MODIFICACION 4: cambia el conjunto de obstaculos activo en tiempo de ejecucion."""
    global OBSTACULOS
    OBSTACULOS = MAPAS[nombre_mapa]

# MODIFICACION 1: el ADN ahora tiene una regla para cada patron de
# 4 bits de obstaculos + 2 bits de direccion relativa a la meta.
# 2^6 = 64 observaciones locales posibles.
NUM_OBSERVACIONES_LOCALES = 64

# MODIFICACION 2: se agrega memoria de la ultima accion ejecutada.
# Puede ser U, D, L, R, o "ninguna" (en el primer paso), o sea 5 estados.
NUM_ESTADOS_MEMORIA = len(COMANDOS) + 1

# El ADN combina ambas cosas: una regla por cada par (observacion local, memoria).
LONGITUD_ADN = NUM_OBSERVACIONES_LOCALES * NUM_ESTADOS_MEMORIA
# MODIFICACION 3: se probaron valores 8, 12 y 40; aqui se deja el
# valor original del laboratorio para las pruebas de la modificacion 4.
PASOS_MAXIMOS = 18
TAMANO_POBLACION = 100
ELITE = 10
PADRES = 40

EMOJI_ROBOT = "\U0001f916"
EMOJI_META = "\U0001f7e9"
EMOJI_META_ALCANZADA = "\U0001f389"
EMOJI_OBSTACULO = "\u2b1b"
EMOJI_INICIO = "\U0001f535"
EMOJI_LIBRE = "\u2b1c"


# ==========================================================
# 2. PERCEPCION Y SIMULACION DEL ENTORNO
# ==========================================================
def limpiar_pantalla():
    """Limpia la terminal antes de dibujar el siguiente paso."""
    os.system("cls" if os.name == "nt" else "clear")


def destino_de(posicion, comando):
    """Devuelve la celda que queda en la direccion indicada."""
    fila, columna = posicion
    cambios = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
    cambio_fila, cambio_columna = cambios[comando]
    return fila + cambio_fila, columna + cambio_columna


def esta_bloqueada(posicion):
    """Indica si una celda no puede ser ocupada por el robot."""
    fila, columna = posicion
    return not (0 <= fila < TAMANO and 0 <= columna < TAMANO) or posicion in OBSTACULOS


def direccion_relativa(posicion):
    """MODIFICACION 1: indica si la meta queda abajo/derecha (1) o arriba/izquierda (0).

    bit_vertical = 1  -> la meta esta en una fila mayor (mas abajo)
    bit_vertical = 0  -> la meta esta en la misma fila o arriba
    bit_horizontal = 1 -> la meta esta en una columna mayor (mas a la derecha)
    bit_horizontal = 0 -> la meta esta en la misma columna o a la izquierda
    """
    fila, columna = posicion
    bit_vertical = 1 if META[0] > fila else 0
    bit_horizontal = 1 if META[1] > columna else 0
    return bit_vertical, bit_horizontal


def observacion_local(posicion):
    """Codifica como bits las celdas bloqueadas (U,D,L,R) mas la direccion a la meta.

    MODIFICACION 1: se agregan 2 bits nuevos al final del patron, con la
    direccion relativa de la meta. El patron completo tiene 6 bits:
    [U][D][L][R][vertical][horizontal] -> 64 valores posibles (0 a 63).
    """
    patron = 0
    for comando in ORDEN_SENSORES:
        patron <<= 1
        patron |= int(esta_bloqueada(destino_de(posicion, comando)))
    bit_vertical, bit_horizontal = direccion_relativa(posicion)
    patron = (patron << 1) | bit_vertical
    patron = (patron << 1) | bit_horizontal
    return patron


def indice_memoria(ultima_accion):
    """MODIFICACION 2: convierte la ultima accion en un indice de 0 a 4.

    0..3 -> el indice de U, D, L o R dentro de COMANDOS.
    4    -> "ninguna", se usa solo en el primer paso del recorrido.
    """
    if ultima_accion is None:
        return len(COMANDOS)
    return COMANDOS.index(ultima_accion)


def observar(posicion, ultima_accion):
    """Combina la observacion local con la memoria de la ultima accion.

    MODIFICACION 2: el indice final ya no es solo el patron de 6 bits;
    se combina con la memoria para obtener un indice unico en
    [0, LONGITUD_ADN). Por ejemplo, la misma vista local con una ultima
    accion distinta selecciona una regla distinta del ADN.
    """
    patron_local = observacion_local(posicion)
    return patron_local * NUM_ESTADOS_MEMORIA + indice_memoria(ultima_accion)


def decidir(adn, posicion, ultima_accion):
    """Consulta la regla del ADN correspondiente a la observacion + memoria actual."""
    return adn[observar(posicion, ultima_accion)]


def mover(posicion, comando):
    """Intenta ejecutar un comando y devuelve posicion y resultado."""
    destino = destino_de(posicion, comando)
    if not (0 <= destino[0] < TAMANO and 0 <= destino[1] < TAMANO):
        return posicion, "borde"
    if destino in OBSTACULOS:
        return posicion, "obstaculo"
    return destino, "avance"


def recorrer(adn):
    """Ejecuta una politica reactiva y registra su trayectoria y metricas.

    MODIFICACION 2: se mantiene ultima_accion durante el recorrido y se
    actualiza despues de cada paso, para que decidir() la use en el
    siguiente paso. Al inicio vale None (todavia no se ejecuto nada).
    """
    posicion = INICIO
    trayectoria = [posicion]
    observaciones = []
    acciones = []
    memorias = []
    choques = 0
    pasos_utiles = 0
    ultima_accion = None
    for _ in range(PASOS_MAXIMOS):
        observacion = observar(posicion, ultima_accion)
        comando = decidir(adn, posicion, ultima_accion)
        nueva_posicion, resultado = mover(posicion, comando)
        observaciones.append(observacion)
        acciones.append(comando)
        memorias.append(ultima_accion)
        if resultado in ("borde", "obstaculo"):
            choques += 1
        if resultado == "avance":
            pasos_utiles += 1
        posicion = nueva_posicion
        trayectoria.append(posicion)
        ultima_accion = comando
        if posicion == META:
            break
    return trayectoria, choques, pasos_utiles, observaciones, acciones, memorias


# ==========================================================
# 3. REGLA DE SUPERVIVENCIA: EL FITNESS
# ==========================================================
def evaluar(adn):
    """Calcula el fitness de una politica: cuanto mayor, mejor solucion."""
    trayectoria, choques, pasos_utiles, _, _, _ = recorrer(adn)
    posicion = trayectoria[-1]
    distancia = abs(META[0] - posicion[0]) + abs(META[1] - posicion[1])
    visitas_repetidas = len(trayectoria) - len(set(trayectoria))
    puntaje = 500
    puntaje -= distancia * 35
    puntaje += pasos_utiles * 4
    puntaje -= choques * 30
    puntaje -= visitas_repetidas * 3
    if posicion == META:
        puntaje += 2000
    return puntaje


def diversidad(poblacion):
    """Cuenta cuantas politicas diferentes hay en la poblacion."""
    return len({"".join(adn) for adn in poblacion})


def reglas_de_politica(adn):
    """Devuelve la politica con sus observaciones como claves legibles.

    MODIFICACION 2: cada indice del ADN combina un patron local (6 bits:
    4 de obstaculos + 2 de direccion a la meta) con la memoria de la
    ultima accion (U, D, L, R o "inicio"). La clave se muestra como
    "patron_local+memoria", por ejemplo "110001+D".
    """
    reglas = {}
    for indice, comando in enumerate(adn):
        patron_local, indice_memo = divmod(indice, NUM_ESTADOS_MEMORIA)
        etiqueta_memoria = (
            COMANDOS[indice_memo] if indice_memo < len(COMANDOS) else "inicio"
        )
        clave = f"{patron_local:06b}+{etiqueta_memoria}"
        reglas[clave] = comando
    return reglas


# ==========================================================
# 4. OPERADORES GENETICOS
# ==========================================================
def mutar(adn, tasa_mutacion, rng):
    """Cambia algunas reglas al azar segun la tasa de mutacion."""
    return [
        rng.choice(COMANDOS) if rng.random() < tasa_mutacion else comando
        for comando in adn
    ]


def cruzar(primer_padre, segundo_padre, rng):
    """Combina el comienzo de una politica y el final de otra."""
    punto = rng.randint(1, len(primer_padre) - 1)
    return primer_padre[:punto] + segundo_padre[punto:]


# ==========================================================
# 5. CICLO EVOLUTIVO
# ==========================================================
def evolucionar(tasa_mutacion, usar_cruce, semilla, maximo_generaciones=3000):
    """Ejecuta el algoritmo genetico y devuelve la mejor politica."""
    rng = random.Random(semilla)
    poblacion = [
        [rng.choice(COMANDOS) for _ in range(LONGITUD_ADN)]
        for _ in range(TAMANO_POBLACION)
    ]
    historial = []
    for generacion in range(maximo_generaciones + 1):
        poblacion.sort(key=evaluar, reverse=True)
        mejor = poblacion[0]
        puntaje = evaluar(mejor)
        trayectoria, choques, pasos_utiles, _, _, _ = recorrer(mejor)
        historial.append((generacion, puntaje, diversidad(poblacion)))
        if trayectoria[-1] == META:
            return resultado(
                usar_cruce, mejor, generacion, puntaje, diversidad(poblacion),
                choques, pasos_utiles, historial, poblacion,
            )
        nueva_poblacion = [robot[:] for robot in poblacion[:ELITE]]
        while len(nueva_poblacion) < TAMANO_POBLACION:
            primer_padre = rng.choice(poblacion[:PADRES])
            if usar_cruce:
                segundo_padre = rng.choice(poblacion[:PADRES])
                hijo = cruzar(primer_padre, segundo_padre, rng)
            else:
                hijo = primer_padre[:]
            nueva_poblacion.append(mutar(hijo, tasa_mutacion, rng))
        poblacion = nueva_poblacion
    poblacion.sort(key=evaluar, reverse=True)
    mejor = poblacion[0]
    trayectoria, choques, pasos_utiles, _, _, _ = recorrer(mejor)
    return resultado(
        usar_cruce, mejor, maximo_generaciones, evaluar(mejor),
        diversidad(poblacion), choques, pasos_utiles, historial, poblacion,
    )


def resultado(usar_cruce, adn, generacion, puntaje, diversidad_final,
              choques, pasos_utiles, historial, poblacion):
    """Reune las metricas que compararemos en el experimento."""
    trayectoria, _, _, _, _, _ = recorrer(adn)
    return {
        "metodo": "mutacion + cruce" if usar_cruce else "solo mutacion",
        "adn": adn,
        "generacion": generacion,
        "puntaje": puntaje,
        "diversidad": diversidad_final,
        "choques": choques,
        "pasos_utiles": pasos_utiles,
        "llego": trayectoria[-1] == META,
        "historial": historial,
        "poblacion": [robot[:] for robot in poblacion],
    }


def guardar_generacion(resultado_actual, identificador, semilla):
    """Guarda la ultima poblacion en un archivo de texto estructurado."""
    identificador_limpio = re.sub(r"[^A-Za-z0-9_.-]+", "_", identificador)
    nombre = f"generacion{resultado_actual['generacion']}_{identificador_limpio}.txt"
    datos = {
        "version": 2,
        "generacion": resultado_actual["generacion"],
        "identificador": identificador_limpio,
        "semilla": semilla,
        "metodo": resultado_actual["metodo"],
        "longitud_adn": LONGITUD_ADN,
        "orden_sensores": "U D L R",
        "poblacion": [
            {
                "indice": indice,
                "adn": "".join(robot),
                "reglas": reglas_de_politica(robot),
            }
            for indice, robot in enumerate(resultado_actual["poblacion"])
        ],
    }
    with open(nombre, "w", encoding="utf-8") as archivo:
        json.dump(datos, archivo, indent=2)
    return nombre


def cargar_generacion(ruta):
    """Carga y valida una poblacion guardada previamente."""
    with open(ruta, encoding="utf-8") as archivo:
        datos = json.load(archivo)
    poblacion = datos.get("poblacion")
    if not isinstance(poblacion, list) or not poblacion:
        raise ValueError("El archivo no contiene una poblacion valida.")
    if datos.get("longitud_adn") != LONGITUD_ADN:
        raise ValueError("La longitud del ADN no coincide con esta version.")
    politicas = [
        individuo["adn"] if isinstance(individuo, dict) else individuo
        for individuo in poblacion
    ]
    if any(
        not isinstance(adn, str)
        or len(adn) != LONGITUD_ADN
        or any(comando not in COMANDOS for comando in adn)
        for adn in politicas
    ):
        raise ValueError("El archivo contiene politicas invalidas.")
    return {
        "generacion": datos.get("generacion", "desconocida"),
        "identificador": datos.get("identificador", os.path.basename(ruta)),
        "metodo": datos.get("metodo", "generacion cargada"),
        "poblacion": [list(adn) for adn in politicas],
    }


def resultado_de_politica(adn, metodo, generacion, poblacion):
    """Construye las metricas necesarias para animar una politica cargada."""
    trayectoria, choques, pasos_utiles, _, _, _ = recorrer(adn)
    return {
        "metodo": metodo,
        "adn": adn,
        "generacion": generacion,
        "puntaje": evaluar(adn),
        "diversidad": diversidad(poblacion),
        "choques": choques,
        "pasos_utiles": pasos_utiles,
        "llego": trayectoria[-1] == META,
        "historial": [],
        "poblacion": [robot[:] for robot in poblacion],
    }


def ejecutar_poblacion(datos, modo_ejecucion):
    """Selecciona el mejor individuo o devuelve todos para animarlos."""
    poblacion = datos["poblacion"]
    poblacion_ordenada = sorted(poblacion, key=evaluar, reverse=True)
    if modo_ejecucion == "mejor":
        return [resultado_de_politica(
            poblacion_ordenada[0], datos["metodo"], datos["generacion"], poblacion
        )]
    return [resultado_de_politica(
        adn, datos["metodo"], datos["generacion"], poblacion
    ) for adn in poblacion]


# ==========================================================
# 6. VISUALIZACION Y CLI
# ==========================================================
def dibujar(resultado, pausa):
    """Anima la trayectoria decidida por la politica reactiva."""
    adn = resultado["adn"]
    trayectoria, _, _, observaciones, acciones, memorias = recorrer(adn)
    for paso, posicion in enumerate(trayectoria[1:], start=1):
        limpiar_pantalla()
        print(
            f"{resultado['metodo'].upper()} | "
            f"Gen {resultado['generacion']} | Puntos: {resultado['puntaje']}"
        )
        print(
            f"{EMOJI_ROBOT} = robot | {EMOJI_META_ALCANZADA} = meta alcanzada | "
            f"{EMOJI_META} = meta | {EMOJI_OBSTACULO} = obstaculo"
        )
        print(f"{EMOJI_INICIO} = inicio | {EMOJI_LIBRE} = celda libre\n")
        for fila in range(TAMANO):
            linea = ""
            for columna in range(TAMANO):
                celda = (fila, columna)
                if celda == posicion:
                    linea += EMOJI_ROBOT
                elif celda == INICIO:
                    linea += EMOJI_INICIO
                elif celda == META:
                    linea += EMOJI_META_ALCANZADA if posicion == META else EMOJI_META
                elif celda in OBSTACULOS:
                    linea += EMOJI_OBSTACULO
                else:
                    linea += EMOJI_LIBRE
            print(linea)
        patron_local, _ = divmod(observaciones[paso - 1], NUM_ESTADOS_MEMORIA)
        memoria_previa = memorias[paso - 1]
        etiqueta_memoria = memoria_previa if memoria_previa else "inicio"
        print(
            f"Paso {paso}/{len(trayectoria) - 1} | "
            f"Observacion local: {patron_local:06b} | "
            f"Memoria previa: {etiqueta_memoria} | "
            f"Accion: {acciones[paso - 1]} | Politica: {''.join(adn)}"
        )
        if pausa:
            time.sleep(pausa)


def mostrar_resultado(resultado, pausa):
    """Muestra la animacion y las metricas finales de una variante."""
    dibujar(resultado, pausa)
    estado = "LLEGO" if resultado["llego"] else "NO LLEGO"
    print(f"\nResultado: {estado}")
    print(f"Pasos utiles: {resultado['pasos_utiles']}")
    print(f"Diversidad final: {resultado['diversidad']}/{TAMANO_POBLACION}")
    if pausa:
        time.sleep(1)


def mostrar_poblacion(resultados, pausa):
    """Anima todos los individuos guardados y muestra un resumen final."""
    for indice, resultado_actual in enumerate(resultados, start=1):
        print(f"\nIndividuo {indice}/{len(resultados)}")
        dibujar(resultado_actual, pausa)
    print(f"\nSe ejecutaron {len(resultados)} individuos en el mapa.")


def main():
    """Lee las opciones de terminal y ejecuta el experimento solicitado."""
    parser = argparse.ArgumentParser(
        description="Compara mutacion y cruce en politicas con percepcion local."
    )
    parser.add_argument(
        "--modo",
        choices=("comparar", "mutacion", "cruce"),
        default="comparar",
    )
    parser.add_argument("--semilla", type=int, default=7)
    parser.add_argument("--sin-pausa", action="store_true")
    parser.add_argument(
        "--identificador",
        help="Texto usado en el nombre generacionN_identificador.txt.",
    )
    parser.add_argument(
        "--cargar",
        metavar="ARCHIVO",
        help="Carga una generacion guardada en lugar de evolucionar.",
    )
    parser.add_argument(
        "--ejecutar",
        choices=("mejor", "todos"),
        default="mejor",
        help="Al cargar, ejecuta el mejor individuo o toda la poblacion.",
    )
    parser.add_argument(
        "--no-guardar",
        action="store_true",
        help="No guarda la poblacion final despues de evolucionar.",
    )
    parser.add_argument(
        "--mapa",
        choices=("original", "alternativo"),
        default="original",
        help="MODIFICACION 4: elige el mapa de obstaculos a usar.",
    )
    args = parser.parse_args()
    usar_mapa(args.mapa)
    pausa = 0 if args.sin_pausa else 0.08

    if args.cargar:
        datos = cargar_generacion(args.cargar)
        resultados = ejecutar_poblacion(datos, args.ejecutar)
        if args.ejecutar == "todos":
            mostrar_poblacion(resultados, pausa)
        else:
            mostrar_resultado(resultados[0], pausa)
        return

    experimentos = {
        "mutacion": ("solo mutacion", False),
        "cruce": ("mutacion + cruce", True),
    }
    modos = tuple(experimentos) if args.modo == "comparar" else (args.modo,)
    resultados = []
    for indice, modo in enumerate(modos):
        _, usar_cruce = experimentos[modo]
        resultado_actual = evolucionar(
            tasa_mutacion=0.08,
            usar_cruce=usar_cruce,
            semilla=args.semilla + indice,
        )
        resultados.append(resultado_actual)
        if not args.no_guardar:
            identificador = args.identificador or (
                f"{modo}_semilla{args.semilla + indice}"
            )
            nombre = guardar_generacion(
                resultado_actual, identificador, args.semilla + indice
            )
            print(f"Generacion guardada en: {nombre}")
        if args.ejecutar == "todos":
            mostrar_poblacion(
                ejecutar_poblacion(
                    {
                        "generacion": resultado_actual["generacion"],
                        "metodo": resultado_actual["metodo"],
                        "poblacion": resultado_actual["poblacion"],
                    },
                    "todos",
                ),
                pausa,
            )
        else:
            mostrar_resultado(resultado_actual, pausa)

    if len(resultados) == 2:
        print("\nCOMPARACION")
        print("Metodo               Llego  Generaciones   Puntaje  Choques  Diversidad")
        print("-----------------------------------------------------------------------")
        for resultado_actual in resultados:
            llego = "si" if resultado_actual["llego"] else "no"
            print(
                f"{resultado_actual['metodo']:<20}{llego:<7}"
                f"{resultado_actual['generacion']:<15}"
                f"{resultado_actual['puntaje']:<9}"
                f"{resultado_actual['choques']:<9}"
                f"{resultado_actual['diversidad']}/{TAMANO_POBLACION}"
            )


if __name__ == "__main__":
    main()