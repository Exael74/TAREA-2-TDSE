# Heart Disease Logistic Regression

Implementación desde cero (NumPy/Pandas/Matplotlib) de regresión logística para predecir presencia de enfermedad cardiaca usando el dataset UCI (303 registros). Incluye EDA, entrenamiento básico, visualización de fronteras, regularización L2 y guía rápida de despliegue en Amazon SageMaker.

## Contenido del repositorio
- `tarea2.ipynb`: notebook principal con EDA, modelo, regularización y notas de despliegue.
- `Heart_Disease_Prediction.csv`: dataset descargado desde Kaggle.
- `best_w.npy`, `best_b.npy`: pesos exportados tras entrenar (se generan al ejecutar el notebook).
- `/images` (pendiente): capturas de SageMaker (entrenamiento, endpoint, inferencia).

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

## Despliegue en Amazon SageMaker (resumen)
- Subir notebook y CSV a SageMaker Studio/Notebook Instance.
- Ejecutar entrenamiento para generar `best_w.npy` y `best_b.npy` (y opcionalmente serializar `feature_mean` y `feature_std`).
- Crear `inference.py` con `model_fn` (carga de pesos) y `predict_fn` (normaliza entrada JSON de 6 features, aplica sigmoid y devuelve probabilidad).
- Empaquetar artefactos y desplegar endpoint vía SDK (`ScriptModel`/`PyTorchModel`).
- Probar con payload ejemplo: `{ "age": 60, "trestbps": 140, "chol": 300, "thalach": 150, "oldpeak": 1.2, "ca": 0 }` → prob ~0.6–0.7.

## Evidencias a capturar (para la entrega)
- Captura del notebook entrenando en SageMaker (job completo).
- Captura del endpoint en estado `InService` con su ARN.
- Captura de una invocación real (respuesta JSON con probabilidad).
- Enlazar repo GitHub público/privado (añadir URL aquí): `https://github.com/<usuario>/heart-disease-lr`.

## Licencia
Uso académico/educativo.
