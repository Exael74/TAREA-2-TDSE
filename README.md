# TAREA-2-TDSE


Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## Autor
- **Nombre del Estudiante** - [STIVEN ESNEIDER PARDO GUTIERREZ](https://github.com/Exael74)

# Heart Disease Logistic Regression

Implementación desde cero (NumPy/Pandas/Matplotlib) de regresión logística para predecir presencia de enfermedad cardiaca usando el dataset UCI (303 registros). Incluye EDA, entrenamiento básico, visualización de fronteras, regularización L2 y guía rápida de despliegue en Amazon SageMaker.

## Contenido del repositorio
- `tarea2.ipynb`: notebook principal con EDA, modelo, regularización y notas de despliegue.
- `Heart_Disease_Prediction.csv`: dataset descargado desde Kaggle.
- `best_w.npy`, `best_b.npy`: pesos exportados tras entrenar (se generan al ejecutar el notebook).
- ![alt text](image.png)


## Dataset
- Fuente: Kaggle UCI Heart Disease — https://www.kaggle.com/datasets/neurocipher/heartdisease
- Tamaño: 303 pacientes, 14 columnas.
- Target binario: 1 = presencia de enfermedad, 0 = ausencia.
- Rango aproximado: edad 29–77, colesterol 112–564 mg/dL; ~55% casos positivos.

## Cómo ejecutar
1. Crear entorno (opcional): `python -m venv .venv` y activar.
2. Instalar dependencias mínimas: `pip install numpy pandas matplotlib`.
3. Abrir `tarea2.ipynb` y ejecutar todas las celdas. Se generarán `best_w.npy` y `best_b.npy`.
4. Para reproducir gráficas/pares de características, asegúrate de que `Heart_Disease_Prediction.csv` está en la misma carpeta.

## Métricas esperadas (ejemplo)
- Accuracy/F1 en train-test alrededor de 0.8–0.9 según la división aleatoria.
- El costo decrece suavemente con α≈0.01 y ~1500 iteraciones.
- La regularización L2 reduce ||w|| y puede mejorar F1 en test dependiendo del lambda.

## Licencia
Uso académico/educativo.


## Notas Adicionales
- Proyecto desarrollado como parte del curso TDSE
- Fecha de entrega: [27/01/2026]
---
