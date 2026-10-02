# Laboratorio IA 8: red neuronal evolutiva con memoria sensorial

Extension del laboratorio 7. La red recibe `s_t`, `s_{t-1}` y `a_{t-1}` (12 entradas)
para romper ciclos en navegacion reactiva. Los cambios respecto al lab 7 estan
marcados con `# LAB 8` en el codigo.

## Archivos
- `robot_red_neuronal_memoria.py`: robot, red, algoritmo genetico y animacion.
- `comparar_lab8.py`: compara red basica (4 entradas) contra red con memoria (12).
- `resultados_lab8/`: tablas (`tablas.md`) y CSV de los experimentos.
- `Reporte_Lab8_Redes_con_Memoria.pdf` y `.docx`: reporte del laboratorio.

## Uso
```bash
pip install -r requirements.txt

# Evolucionar con memoria y guardar
python3 robot_red_neuronal_memoria.py --semilla 7 --mapa mapa2 --guardar-mejor mejor_red.pt

# Red basica para comparar
python3 robot_red_neuronal_memoria.py --basica --semilla 7

# Probar la misma red en otro mapa sin re-evolucionar
python3 robot_red_neuronal_memoria.py --cargar-red mejor_red.pt --mapa pasillos

# Tablas completas
python3 comparar_lab8.py --semillas 30 --salida resultados_lab8
```

Mapas disponibles: `sencillo`, `mapa2`, `pasillos`, `bucles`.
Opciones utiles: `--sin-pausa`, `--sin-animacion`, `--generaciones N`.
