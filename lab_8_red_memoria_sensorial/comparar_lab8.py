"""LAB 8: compara la red basica (4 entradas) contra la red con memoria (12).

Experimento A (entrenar y evaluar en el mismo mapa):
    para cada mapa y cada tipo de red, se evoluciona con N semillas y se mide
    el mejor individuo de cada corrida en ese mismo mapa.

Experimento B (generalizacion, seccion 8 del enunciado):
    se evoluciona SOLO en el mapa de entrenamiento y se prueba la misma red,
    sin tocar pesos, en los demas mapas.

Uso:
    python3 comparar_lab8.py --semillas 30 --salida resultados_lab8
"""
import argparse
import csv
import os
from statistics import mean

import robot_red_neuronal_memoria as lab

TIPOS = {"basica": False, "memoria": True}


def resumir(filas):
    n = len(filas)
    return {
        "exito_%": 100 * sum(f["exito"] for f in filas) / n,
        "choques": mean(f["choques"] for f in filas),
        "pasos": mean(f["pasos"] for f in filas),
        "ciclos": mean(f["ciclos"] for f in filas),
        "dist_final": mean(f["distancia_final"] for f in filas),
    }


def tabla_md(titulo, columnas_clave, filas):
    lineas = [f"### {titulo}", ""]
    cab = columnas_clave + ["red", "exito_%", "choques", "pasos", "ciclos", "dist_final"]
    lineas.append("| " + " | ".join(cab) + " |")
    lineas.append("|" + "|".join(["---"] * len(cab)) + "|")
    for f in filas:
        vals = [str(f[c]) for c in columnas_clave] + [f["red"]]
        vals += [f"{f['exito_%']:.0f}", f"{f['choques']:.2f}", f"{f['pasos']:.2f}", f"{f['ciclos']:.2f}", f"{f['dist_final']:.2f}"]
        lineas.append("| " + " | ".join(vals) + " |")
    return "\n".join(lineas) + "\n"


def experimento_a(semillas, generaciones):
    salida = []
    for mapa, obstaculos in lab.MAPAS.items():
        for tipo, usar_memoria in TIPOS.items():
            filas = []
            for s in semillas:
                r = lab.evolucionar(semilla=s, maximo_generaciones=generaciones,
                                    usar_memoria=usar_memoria, obstaculos=obstaculos)
                filas.append(lab.metricas(r["red"], obstaculos=obstaculos))
            salida.append({"mapa": mapa, "red": tipo, **resumir(filas)})
    return salida


def experimento_b(semillas, generaciones, mapa_train):
    salida = []
    for tipo, usar_memoria in TIPOS.items():
        redes = [
            lab.evolucionar(semilla=s, maximo_generaciones=generaciones,
                            usar_memoria=usar_memoria, obstaculos=lab.MAPAS[mapa_train])["red"]
            for s in semillas
        ]
        for mapa, obstaculos in lab.MAPAS.items():
            if mapa == mapa_train:
                continue
            filas = [lab.metricas(red, obstaculos=obstaculos) for red in redes]
            salida.append({"entrenada_en": mapa_train, "probada_en": mapa, "red": tipo, **resumir(filas)})
    salida.sort(key=lambda f: (f["probada_en"], f["red"]))
    return salida


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--semillas", type=int, default=30)
    p.add_argument("--generaciones", type=int, default=40)
    p.add_argument("--entrenar-en", type=str, default="sencillo", choices=sorted(lab.MAPAS))
    p.add_argument("--salida", type=str, default="resultados_lab8")
    args = p.parse_args()

    semillas = list(range(1, args.semillas + 1))
    os.makedirs(args.salida, exist_ok=True)

    a = experimento_a(semillas, args.generaciones)
    b = experimento_b(semillas, args.generaciones, args.entrenar_en)

    md = tabla_md(f"Experimento A: entrenar y evaluar en el mismo mapa ({args.semillas} semillas)", ["mapa"], a)
    md += "\n" + tabla_md(f"Experimento B: entrenada en '{args.entrenar_en}', probada sin re-evolucionar", ["entrenada_en", "probada_en"], b)
    print(md)

    with open(os.path.join(args.salida, "tablas.md"), "w", encoding="utf-8") as f:
        f.write(md)
    for nombre, datos in (("experimento_a", a), ("experimento_b", b)):
        with open(os.path.join(args.salida, f"{nombre}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(datos[0].keys()))
            w.writeheader()
            w.writerows(datos)


if __name__ == "__main__":
    main()
