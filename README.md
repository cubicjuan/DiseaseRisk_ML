
# Análisis de Salud con Machine Learning

Este proyecto tiene como objetivo realizar un análisis de datos de salud para predecir si una persona se encuentra saludable o no, utilizando técnicas de Machine Learning. A través de un análisis exploratorio de datos (EDA), exploramos la relación entre diferentes variables que podrían influir en el estado de salud de las personas.

## Descripción del Proyecto

El proyecto está basado en un conjunto de datos que contiene varias variables relacionadas con el bienestar físico y mental de las personas. El objetivo es utilizar estas variables para predecir si una persona se considera "saludable" o no. Durante el análisis, se realizan diversas visualizaciones para explorar la distribución de los datos y entender las relaciones entre las variables.

### Variables Disponibles

El conjunto de datos contiene las siguientes variables:

1. **age**: Edad de la persona (numérica).
2. **gender**: Género de la persona (categórica: 'M' para masculino, 'F' para femenino).
3. **height**: Altura de la persona (numérica, en metros).
4. **weight**: Peso de la persona (numérica, en kilogramos).
5. **bmi**: Índice de Masa Corporal (numérica).
6. **waist_size**: Tamaño de cintura (numérica).
7. **blood_pressure**: Presión arterial (numérica).
8. **heart_rate**: Frecuencia cardíaca (numérica).
9. **cholesterol**: Nivel de colesterol (numérica).
10. **glucose**: Nivel de glucosa (numérica).
11. **sleep_hours**: Horas de sueño por noche (numérica).
12. **sleep_quality**: Calidad del sueño (numérica).
13. **work_hours**: Número de horas de trabajo diarias (numérica).
14. **physical_activity**: Nivel de actividad física (numérica).
15. **daily_steps**: Número de pasos dados al día (numérica).
16. **calorie_intake**: Ingesta calórica diaria (numérica).
17. **sugar_intake**: Ingesta de azúcar diaria (numérica).
18. **alcohol_consumption**: Consumo de alcohol (numérica).
19. **smoking_level**: Nivel de tabaquismo (numérica).
20. **water_intake**: Ingesta de agua diaria (numérica).
21. **screen_time**: Tiempo frente a la pantalla (numérica).
22. **stress_level**: Nivel de estrés (numérica).
23. **mental_health_score**: Puntuación de salud mental (numérica).
24. **education_level**: Nivel educativo (categórica).
25. **job_type**: Tipo de trabajo (categórica).
26. **income**: Nivel de ingresos (numérica).
27. **diet_type**: Tipo de dieta seguida (categórica).
28. **exercise_type**: Tipo de ejercicio practicado (categórica).
29. **device_usage**: Uso de dispositivos (numérica).
30. **healthcare_access**: Acceso a atención médica (categórica).
31. **insurance**: Póliza de seguro médico (categórica).
32. **sunlight_exposure**: Exposición al sol (numérica).
33. **meals_per_day**: Número de comidas al día (numérica).
34. **family_history**: Historial familiar de enfermedades (categórica).
35. **pet_owner**: Propietario de mascota (categórica).
36. **electrolyte_level**: Nivel de electrolitos en el cuerpo (numérica).
37. **gender_marker_flag**: Indicador de género (categórica).
38. **environmental_risk_score**: Puntuación de riesgo ambiental (numérica).
39. **daily_supplement_dosage**: Dosificación diaria de suplementos (numérica).

### Objetivo

El principal objetivo es predecir si una persona es saludable o no basándonos en las variables disponibles. Esto se realizará utilizando técnicas de **Machine Learning**, entrenando modelos sobre los datos y evaluando su rendimiento.

## Análisis Exploratorio de Datos (EDA)

Antes de construir el modelo de predicción, se realizará un análisis exploratorio de los datos para entender las relaciones entre las variables y su distribución. 

## Selección de variables

1. **Preprocesamiento**: Prepararemos los datos para el modelo de Machine Learning. Esto incluye la normalización de variables, la codificación de variables categóricas y la imputación de valores faltantes.
2. **Revisión de varias técnicas**: Se trabajan las diferentes técnicas para conocer cual es la mejor forma para disminuir la cantidad de variables y se explique el 90%. Se revisan técnicas como Filtrado, incrustado, envoltura, PCA y MCA.
3. **PCA y MCA**: Se escogen las técnicas de PCA para las variables númericas y MCA para las variables categoricas en la reducción de dimensionalidad. Generando un conjunto de datos preparados para la aplicación de Machine Learning.
4. **Modelado**: Se revisan los diferentes modelos con los datos establecidos. Se revisan modelos como Random Forest, Gradient Boosting, HistGradien Boosting, Extra Trees y Decision Trees.
5. **Evaluación del Modelo**: Evaluaremos el rendimiento de los modelos utilizando métricas como **precisión**, **recall** y **F1-score**.

## Requisitos

Este proyecto requiere las siguientes bibliotecas de Python:

### `numpy`
- **Descripción**: `numpy` es la biblioteca fundamental para realizar cálculos numéricos en Python. Proporciona soporte para arrays multidimensionales (`ndarray`) y funciones matemáticas eficientes para trabajar con grandes volúmenes de datos.
- **Instalación**: `pip install numpy`

### `scipy`
- **Descripción**: `scipy` se utiliza para resolver problemas científicos y técnicos complejos. Basado en `numpy`, incluye herramientas para álgebra lineal, optimización, estadísticas, y resolución de ecuaciones diferenciales.
- **Instalación**: `pip install scipy`

### `seaborn`
- **Descripción**: `seaborn` es una librería de visualización de datos basada en `matplotlib`. Facilita la creación de gráficos estadísticos atractivos y complejos con menos líneas de código.
- **Instalación**: `pip install seaborn`

### `imbalanced-learn`
- **Descripción**: `imbalanced-learn` es útil para tratar datasets desbalanceados, proporcionando técnicas como sobremuestreo, submuestreo y SMOTE (Synthetic Minority Over-sampling Technique) para mejorar el rendimiento de los modelos.
- **Instalación**: `pip install imbalanced-learn`

### `prince`
- **Descripción**: `prince` es una librería para realizar técnicas de reducción dimensional como el Análisis de Componentes Principales (PCA), el Análisis de Correspondencias (CA) y el Escalado Multidimensional (MDS).
- **Instalación**: `pip install prince`

### `mca`
- **Descripción**: `mca` se utiliza para realizar el Análisis de Correspondencias Múltiples (MCA), que es útil para analizar datos categóricos y reducir la dimensionalidad de este tipo de datos.
- **Instalación**: `pip install mca`

Puedes instalar las dependencias ejecutando:

```bash
pip install pandas matplotlib seaborn scikit-learn numpy scipy seaborn imbalanced-learn prince mca
```

## Cómo Correr el Proyecto

1. Clona este repositorio en tu máquina local:
   ```bash
   git clone https://github.com/cubicjuan/DiseaseRisk_ML
   ```
2. Navega al directorio del proyecto:
   ```bash
   cd proyecto-salud-ml
   ```
3. Corre el script de análisis y modelado:
   ```bash
   Trabajofinal.py
   ```

## Contribuciones

Las contribuciones a este proyecto son bienvenidas. Si tienes alguna sugerencia o mejora, siéntete libre de abrir un **Issue** o enviar un **Pull Request**.

---

**Autor**: Juan Francisco Reyes y Mónica Cristancho Ferrer  
**Fecha**: Agosto 2025  
