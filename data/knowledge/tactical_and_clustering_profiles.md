# Taxonomía de Clubes y Perfiles No Supervisados (PCA y K-Means)

## 1. Fundamento del Análisis No Supervisado
Para superar la simple lectura posicional de la tabla, se construyó una matriz multidimensional estandarizada con 7 métricas de rendimiento por equipo-temporada (60 observaciones en total):
- Puntos totales acumulados.
- Goles a favor y goles en contra.
- Diferencia de goles.
- Conteo de victorias, empates y derrotas.

## 2. Reducción Dimensional mediante PCA
- **PC1 (77.31% de varianza explicada):** "Eje de Dominancia Competitiva". Proyecta la jerarquía general del equipo. Correlaciona fuertemente con puntos (+0.43), diferencia de goles (+0.43), victorias (+0.42) y goles a favor (+0.38), y de forma inversa con derrotas (-0.42).
- **PC2 (16.98% de varianza explicada):** "Eje de Estilo y Propensión al Empate". Correlaciona fuertemente con empates (+0.90) y de forma negativa con goles a favor y en contra (-0.24), separando a equipos pragmáticos/cerrados de aquellos con partidos de alta volatilidad.
- **Varianza Acumulada 2D:** 94.29% de la información total se conserva en el plano bidimensional.

## 3. Clústeres K-Means y Justificación Analítica de k=3
Aunque el coeficiente de silueta alcanza 0.4012 en k=2 frente a 0.3206 en k=3, la selección de k=3 responde a una decisión analítica rigurosa:
1. **Inflexión del Codo:** La inercia experimenta su caída marginal más pronunciada al pasar de k=2 a k=3 (de 208.52 a 152.49, caída de 56.03 unidades).
2. **Realidad de la Liga:** k=2 colapsa la liga en una dicotomía binaria ("buenos" vs "malos"), invisibilizando a la clase media consolidada.
3. **Estabilidad Estocástica:** K-Means con k=3 alcanza un Adjusted Rand Index (ARI) de 1.0000 en 10 semillas aleatorias independientes.

## 4. Descripción de los Tres Estratos Competitivos
- **Clúster 0 — Alto Rendimiento / Contendientes Europeos (21 perfiles):** Promedian 72.5 puntos, +29.2 de diferencia de goles y 21.5 victorias. Integrado por clubes clasificados a Champions League, Europa League y Conference League (Manchester City, Arsenal, Liverpool, Manchester United, Newcastle, Brighton '22, Aston Villa '22).
- **Clúster 2 — Mitad de Tabla Consolidada (29 perfiles):** Promedian 47.7 puntos, -7.0 de diferencia de goles y 12.7 victorias. Equipos que garantizan permanencia sin pelear el campeonato: Fulham, Brentford, Crystal Palace, Brighton, West Ham.
- **Clúster 1 — Zona de Descenso y Permanencia (10 perfiles):** Promedian 25.4 puntos, -41.0 de diferencia de goles y 24.4 derrotas. Clubes con alta fragilidad defensiva que luchan o sufren el descenso directo (Sheffield United, Luton Town, Burnley, Southampton).
