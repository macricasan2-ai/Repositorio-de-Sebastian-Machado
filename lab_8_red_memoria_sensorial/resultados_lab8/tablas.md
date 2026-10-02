### Experimento A: entrenar y evaluar en el mismo mapa (30 semillas)

| mapa | red | exito_% | choques | pasos | ciclos | dist_final |
|---|---|---|---|---|---|---|
| sencillo | basica | 97 | 0.00 | 14.60 | 0.80 | 0.40 |
| sencillo | memoria | 100 | 2.77 | 19.83 | 0.00 | 0.00 |
| mapa2 | basica | 97 | 0.00 | 14.60 | 0.60 | 0.13 |
| mapa2 | memoria | 100 | 2.87 | 17.60 | 0.00 | 0.00 |
| pasillos | basica | 0 | 0.00 | 30.00 | 17.97 | 5.13 |
| pasillos | memoria | 13 | 1.00 | 29.73 | 12.27 | 3.33 |
| bucles | basica | 100 | 0.00 | 14.00 | 0.00 | 0.00 |
| bucles | memoria | 100 | 2.07 | 16.60 | 0.00 | 0.00 |

### Experimento B: entrenada en 'sencillo', probada sin re-evolucionar

| entrenada_en | probada_en | red | exito_% | choques | pasos | ciclos | dist_final |
|---|---|---|---|---|---|---|---|
| sencillo | bucles | basica | 97 | 0.00 | 14.53 | 0.60 | 0.27 |
| sencillo | bucles | memoria | 73 | 4.67 | 22.30 | 2.90 | 1.37 |
| sencillo | mapa2 | basica | 93 | 0.00 | 15.13 | 1.60 | 0.73 |
| sencillo | mapa2 | memoria | 57 | 10.37 | 24.40 | 9.87 | 4.23 |
| sencillo | pasillos | basica | 0 | 16.07 | 30.00 | 19.33 | 4.53 |
| sencillo | pasillos | memoria | 0 | 21.43 | 30.00 | 19.77 | 7.83 |

### Experimento C: mapa 'pasillos' con 150 generaciones (10 semillas)

| red | exito_% | ciclos | dist_final |
|---|---|---|---|
| basica | 0 | 16.6 | 4.40 |
| memoria | 70 | 3.1 | 0.90 |
