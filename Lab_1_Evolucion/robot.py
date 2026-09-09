import random
import time
import os

def limpiar_pantalla():
    os.system('cls' if os.name == 'nt' else 'clear')

# ==========================================================
# ZONA DE TRABAJO (LA REGLA DE SUPERVIVENCIA)
# ==========================================================
def evaluar_robot(adn):
    posicion = [0, 0]  # Coordenadas iniciales: [Fila, Columna]
    acido = [4, 4]      # Coordenada del pozo de acido letal
    piso_acido = False   # Bandera para saber si el robot cayo en el acido

    # El robot ejecuta su secuencia genetica a ciegas
    for comando in adn:
        if comando == 'U': posicion[0] -= 1
        elif comando == 'D': posicion[0] += 1
        elif comando == 'L': posicion[1] -= 1
        elif comando == 'R': posicion[1] += 1

        # Revisamos en CADA paso si el robot piso el acido
        if posicion == acido:
            piso_acido = True

    cubo = [7, 7]  # Coordenadas de la meta
    distancia = abs(cubo[0] - posicion[0]) + abs(cubo[1] - posicion[1])

    # El fitness base premia la proximidad a la meta
    puntaje = 100 - distancia

    # Penalizacion masiva si el robot piso el acido en algun punto del camino
    if piso_acido:
        puntaje -= 50

    return puntaje

# ==========================================================
# MOTOR GRAFICO (NO MODIFICAR)
# ==========================================================
def animar_mejor_robot(adn, generacion, puntaje):
    posicion, cubo, tamano = [0, 0], [7, 7], 8
    for paso, comando in enumerate(adn):
        limpiar_pantalla()
        print(f"Gen {generacion} | Puntos: {puntaje}/100")
        if comando == 'U': posicion[0] -= 1
        elif comando == 'D': posicion[0] += 1
        elif comando == 'L': posicion[1] -= 1
        elif comando == 'R': posicion[1] += 1
        posicion[0] = max(0, min(posicion[0], tamano - 1))
        posicion[1] = max(0, min(posicion[1], tamano - 1))
        for fila in range(tamano):
            linea = ""
            for col in range(tamano):
                if fila == cubo[0] and col == cubo[1]:
                    linea += "[META]" if posicion == cubo else "[CUBO]"
                elif fila == posicion[0] and col == posicion[1]:
                    linea += "[ROBOT]"
                else:
                    linea += "[ ]"
            print(linea)
        time.sleep(0.08)

# ==========================================================
# CICLO EVOLUTIVO (NO MODIFICAR)
# ==========================================================
comandos = ['U', 'D', 'L', 'R']
mejor_robot = "".join(random.choice(comandos) for _ in range(30))
generacion = 0

while evaluar_robot(mejor_robot) < 100:
    robot_mutante = ""
    for gen in mejor_robot:
        robot_mutante += random.choice(comandos) if random.random() < 0.15 else gen
    if evaluar_robot(robot_mutante) > evaluar_robot(mejor_robot):
        mejor_robot = robot_mutante
    generacion += 1
    # Se anima el progreso cada 150 iteraciones
    if generacion % 150 == 0:
        animar_mejor_robot(mejor_robot, generacion, evaluar_robot(mejor_robot))

animar_mejor_robot(mejor_robot, generacion, evaluar_robot(mejor_robot))