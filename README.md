# TMF_IA

Monitorización de sensores en fábricas con redes neuronales recurrentes (RNN).

## Descripción del Proyecto

Este proyecto se enfoca en la monitorización de sensores en fábricas utilizando redes neuronales recurrentes (RNN). El objetivo es capturar y analizar datos de sensores para mejorar la eficiencia y detectar posibles fallos en tiempo real.

## Archivos del Proyecto

### `api_server_RPi.py`

Este archivo contiene el código en Python para simular en un Raspberry Pi 4 la señal de 2 sensores y enviar los datos por MQTT a una plataforma de monitoreo llamada ThingsBoard.

### `train_TFM.ipynb`

Este notebook se utiliza para traer la data simulada desde la plataforma IoT y entrenar varias configuraciones con modelos LSTM y GRU para predecir las señales.

### `inferencia_TFM.ipynb`

Este notebook permite hacer inferencia con los mejores modelos guardados para cada señal y generar alertas en la plataforma IoT en base a umbrales definidos.

## Directorios del Proyecto

### `data/`

Este directorio contiene la telemetría de las variables a trabajar (Corriente y Vibración).

### `modelos/`

Este directorio almacena los mejores 2 modelos de cada característica entrenada y los archivos de scaler de cada característica (.pkl) creados durante el entrenamiento para usar en la inferencia.

## Requisitos

- Raspberry Pi 4
- Python 3.x
- MQTT Broker
- Plataforma de Monitoreo ThingsBoard

## Instalación

1. Clonar este repositorio:
    ```bash
    git clone https://github.com/rabsedna/TMF_IA2.git
    ```

2. Instalar las dependencias necesarias:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

1. Ejecutar el script `api_server_RPi.py` para iniciar la simulación y el envío de datos de los sensores:
    ```bash
    python api_server_RPi.py
    ```

2. Utilizar `train_TFM.ipynb` para entrenar los modelos LSTM y GRU con los datos simulados desde la plataforma IoT.

3. Utilizar `inferencia_TFM.ipynb` para hacer inferencia con los mejores modelos y generar alertas en la plataforma IoT.

4. Asegurarse que la plataforma ThingsBoard esté configurada para recibir y visualizar los datos enviados.

## Contribución

Las contribuciones son bienvenidas. Por favor, abre un issue o envía un pull request para discutir cualquier cambio que desees realizar.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.