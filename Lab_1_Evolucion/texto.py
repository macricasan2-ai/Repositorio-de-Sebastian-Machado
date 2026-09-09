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