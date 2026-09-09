import random
import time
import string
import os

def limpiar_pantalla():
    os.system('cls' if os.name == 'nt' else 'clear')

# El entorno (La palabra a evolucionar)
# REGLA DE ORO: Solo mayusculas y espacios
objetivo = "SEBASTIAN MACHADO PARRA"
letras_posibles = string.ascii_uppercase + " "

# El azar inicial (ADN basura)
individuo = "".join(random.choice(letras_posibles) for _ in range(len(objetivo)))
generacion = 0

# El motor evolutivo
while individuo != objetivo:
    nuevo_individuo = ""
    for i in range(len(objetivo)):
        # Seleccion natural a nivel de caracter
        if individuo[i] == objetivo[i]:
            nuevo_individuo += individuo[i]
        else:
            nuevo_individuo += random.choice(letras_posibles)
    individuo = nuevo_individuo
    generacion += 1
    limpiar_pantalla()
    print(f"Gen {generacion}: {individuo}")
    time.sleep(0.05)

print(f"\nEvolucion completada en {generacion} generaciones!")