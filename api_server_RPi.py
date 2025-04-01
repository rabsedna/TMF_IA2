import numpy as np
import paho.mqtt.client as mqtt
import json
import time

# -------------------- Configuración de la Simulación --------------------
# La clase SignalConfig contiene los parámetros de configuración para cada señal.
class SignalConfig:
    def __init__(self,
                 base=None,
                 noise=None,
                 fault_start=None,
                 apply_fault=False,
                 frequency=None,
                 vib_failure_factor=None,
                 fluctuation=None,
                 rate=None,
                 f_inc=None,
                 dip_prob=None):
        self.base = base                      # Valor base de la señal (amplitud o media base de las señal)
        self.noise = noise                    # Nivel de ruido agregado a la señal
        self.fault_start = fault_start        # Tiempo en el que se activa el fallo (solo cuando se usa por tiempo)
        self.apply_fault = apply_fault        # Indicador para aplicar el fallo
        self.frequency = frequency            # Frecuencia principal de la señal (vibración o para cálculos en corriente)
        self.vib_failure_factor = vib_failure_factor if vib_failure_factor is not None else 0.5  # Factor de fallo para vibración
        self.fluctuation = fluctuation        # Parámetro de fluctuación (utilizado en corriente)
        self.rate = rate                      # Tasa para el incremento exponencial en caso de fallo (corriente)
        self.f_inc = f_inc                    # Incremento máximo relativo en caso de fallo (corriente)
        self.dip_prob = dip_prob              # Probabilidad de ocurrencia de un dip (bajón) en la señal de corriente

# -------------------- Función de Simulación de Vibración --------------------
def simulate_vibration(t, config, sg_counter, fault_init):
    """
    Simula una señal de vibración compleja compuesta por:
      - Un ciclo global (mayor) que modula la intensidad de la vibración a lo largo del tiempo.
      - Ciclos internos rápidos que representan la vibración base.
      - Picos o bursts cíclicos y repentinos dentro de los ciclos mayores.
      - Ruido gaussiano que añade variabilidad sutil.
      - Un efecto de fallo opcional (vibración adicional de mayor frecuencia) activado después de fault_start.
      
    La combinación de estos elementos permite que en algunos momentos la vibración sea casi nula y en otros se incremente de forma predecible.
    """
    # Parámetros base
    amplitude = config.base if config.base is not None else 1.0
    freq = config.frequency if config.frequency is not None else 50
    noise_level = config.noise if config.noise is not None else 0.2

    # 1. Ciclo global mayor: modula la amplitud a lo largo del tiempo
    #    Utilizamos un periodo largo (por ejemplo, 30 segundos) para simular fases de alta y baja vibración.
    periodo_global = 30  # segundos
    envelope_global = (np.sin(2 * np.pi * (1 / periodo_global) * t) + 1) / 2  
    # envelope_global varía entre 0 (casi sin vibración) y 1 (máxima vibración)

    # 2. Ciclos internos: vibración base rápida, utilizando la frecuencia principal
    internal_component = np.sin(2 * np.pi * freq * t)

    # 3. Componente de bursts o picos cíclicos:
    #    Se generan picos: un burst_period, con duración burst_duration.
    burst_period = 10    # segundos: cada 10 s se produce un burst
    burst_duration = 2   # segundos: la duración de cada burst
    burst_mask = (np.mod(t, burst_period) < burst_duration)
    burst_amplitude = amplitude * 0.8  # amplitud del burst (ajustable)
    burst_component = np.zeros(len(t))
    # Los bursts se generan con una frecuencia algo mayor (por ejemplo, 1.5 veces la frecuencia base)
    burst_component[burst_mask] = burst_amplitude * np.sin(2 * np.pi * (freq * 1.5) * t[burst_mask])

    # 4. Combinación de los componentes:
    #    Se utiliza el envelope_global para modular la vibración interna,
    #    y se suma la componente de bursts que añade picos repentinos.
    signal = envelope_global * amplitude * internal_component + burst_component

    # 5. Agregar ruido gaussiano sutil que simula vibraciones leves aleatorias
    noise = np.random.normal(0, noise_level, len(t))

    # 6. Efecto de fallo: se añade un componente adicional de mayor frecuencia (por ejemplo, doble frecuencia)
    fault_effect = np.zeros(len(t))
    if config.apply_fault or sg_counter >= config.fault_start:
        fault_idx = t >= fault_init
        fault_effect[fault_idx] = config.vib_failure_factor * amplitude * np.sin(2 * np.pi * (freq * 2) * t[fault_idx])

    # Se retorna la suma de la señal modulada, el ruido y el efecto de fallo (si está activado)
    return signal + noise + fault_effect


# -------------------- Función de Simulación de Corriente --------------------
def simulate_current(t, config, sg_counter, fault_init):
    """
    Simula una señal de corriente compuesta por:
      1. Rampa inicial: Durante los primeros 60 segundos, la corriente incrementa gradualmente (x ej. +2 A).
      2. Fluctuación cíclica: Ciclos de variación de baja frecuencia (aprox. 2.5 minutos) con amplitud ~1 A.
      3. Ruido leve: Se suma ruido gaussiano a la señal.
      4. Dips suaves: Bajones en la señal que ocurren con transición gradual (3 a 5 segundos).
      5. Efecto de fallo: Incremento adicional exponencial si se activa el fallo.
      6. Caídas periódicas: Cada 5 minutos se genera una caída suave de la corriente hasta casi 0 o 1 A,
         simulando momentos en que la máquina se detiene para iniciar un nuevo ciclo.
    """
    # Parámetros básicos
    base_current = config.base if config.base is not None else 10
    noise_level  = config.noise if config.noise is not None else 0.05
    fs = 15  # Frecuencia de muestreo (para calcular los dips)
    t = np.array(t)  # Asegurarse de que t es un array

    # 1. Rampa inicial: Incremento lineal en los primeros 60 segundos
    delta_ramp = 2.0  # Incremento total de 2 A en 60 segundos
    ramp = np.where(t < 60, delta_ramp * (t / 60), delta_ramp)

    # 2. Fluctuación cíclica: Variación sinusoidal a partir de t >= 60 s
    cycle_amplitude = 1.0    # Amplitud del ciclo (1 A)
    cycle_period = 150       # Periodo del ciclo (150 s, aprox. 2.5 minutos)
    fluctuation_component = np.zeros_like(t)
    mask_post60 = t >= 60
    fluctuation_component[mask_post60] = cycle_amplitude * np.sin(2 * np.pi * (t[mask_post60] - 60) / cycle_period)

    # 3. Señal base: Suma del valor base, la rampa y la fluctuación
    current_signal = base_current + ramp + fluctuation_component

    # 4. Agregar ruido leve
    noise = np.random.normal(0, noise_level, len(t))
    current_signal += noise

    # 5. Generación de dips suaves: Bajones con transición gradual (3 a 5 segundos)
    dip_probability = config.dip_prob if config.dip_prob is not None else 0.005
    current_envelope = np.ones_like(t)
    i = 0
    while i < len(t):
        if np.random.rand() < dip_probability:
            dip_duration = int(np.random.uniform(3, 5) * fs)  # Duración en muestras
            dip_depth = 0.7  # La corriente cae al 30% de su valor normal
            # Ventana de Hann para una transición suave
            window = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, dip_duration)))
            smooth_dip = 1 - dip_depth * window
            end_idx = min(i + dip_duration, len(t))
            current_envelope[i:end_idx] *= smooth_dip[:end_idx - i]
            i += dip_duration
        else:
            i += 1
    current_signal *= current_envelope

    # 6. Efecto de fallo: Incremento adicional exponencial en la corriente
    if config.apply_fault or sg_counter >= config.fault_start:
        fault_idx = t >= fault_init
        time_fault = t[fault_idx] - fault_init
        max_incr = base_current * config.f_inc  # incremento máximo (p.ej., 50% de base)
        exp_incr = max_incr * (1 - np.exp(-config.rate * time_fault))
        # Pico aleatorio: en cada muestra, con probabilidad 5, se suma un pico entre 0 y max_incr
        peaks = np.where(np.random.rand(len(time_fault)) < 0.05, np.random.uniform(0, max_incr, len(time_fault)), 0)
        current_signal[fault_idx] += exp_incr + peaks

    #if config.apply_fault or sg_counter >= config.fault_start:
    #    fault_idx = t >= fault_init
    #    max_increase = base_current * (config.f_inc if config.f_inc is not None else 0.07)
    #    incremental_increase = max_increase * (1 - np.exp(-config.rate * (t[fault_idx] - fault_init)))
    #    current_signal[fault_idx] += incremental_increase

    # 7. Caídas periódicas: Cada 5 minutos la corriente cae suavemente hasta casi "0 o 1 A"
    stop_period = 300        # Cada 300 segundos (5 minutos)
    drop_duration = 10.0     # Duración del evento de caída (10 s)
    drop_value = 1.0         # Valor de corriente durante la caída (puede ser cercano a 0 o 1 A)
    # Para cada instante se calcula el tiempo transcurrido dentro del ciclo de 5 minutos
    m = np.mod(t, stop_period)
    blend = np.zeros_like(t)
    mask = m < drop_duration  # Indices dentro del evento de caída
    # Usamos una función cosenoidal para una transición suave:
    # blend = 0 en el inicio y fin del evento, y alcanza 1 en el centro
    blend[mask] = 0.5 * (1 - np.cos(2 * np.pi * m[mask] / drop_duration))
    # Mezclamos la señal calculada con el valor de caída
    current_signal = (1 - blend) * current_signal + blend * drop_value

    return current_signal


# -------------------- Función para Activar el Fallo --------------------
def activate_fault(config_list):
    """
    Activa el efecto de fallo en todas las señales.
    """
    for config in config_list:
        config.apply_fault = True
    print("Fallo activado en todas las señales.")

# -------------------- Función para Enviar Datos Continuos vía MQTT --------------------
def simulate_signals_continuous(vib_config, curr_config, mqtt_client, telemetry_topic, fs, duration, fault_init):
    """
    Genera y envía datos continuos de vibración y corriente.
    Se publican los valores de cada señal vía MQTT en el tópico indicado.
    """
    t0 = time.time()
    dt = 1.0 / fs
    num_samples = int(duration * fs)
    
    for i in range(num_samples):
        t_now = time.time() - t0
        t_sample = np.array([t_now])
        
        # Simula la vibración y la corriente
        sg_counter = i/fs
        vibration = simulate_vibration(t_sample, vib_config, sg_counter, fault_init)[0]
        currentVal = simulate_current(t_sample, curr_config, sg_counter, fault_init)[0]
        
        # Construir payload para MQTT
        payload = {
            "Vibracion": round(vibration, 2),
            "Corriente": round(currentVal, 2)
        }
        
        mqtt_client.publish(telemetry_topic, json.dumps(payload))
        print(f"Publicado: {payload}")
        time.sleep(dt)
    
    print("Envío continuo finalizado.")

# -------------------- Configuración de MQTT --------------------
mqttServer = "192.168.5.110"
mqttPort = 1883
accessToken = "jv26OEmBl5d6XW8mPxVo"
telemetryTopic = "v1/devices/me/telemetry"

def on_connect(client, userdata, flags, rc):
    """
    Callback de conexión: informa si la conexión con el broker MQTT fue exitosa.
    """
    if rc == 0:
        print("Conexión exitosa con el broker MQTT.")
    else:
        print("Error en la conexión. Código de resultado: " + str(rc))

# Configuración del cliente MQTT
mqtt_client = mqtt.Client()
mqtt_client.username_pw_set(accessToken)
mqtt_client.on_connect = on_connect
mqtt_client.connect(mqttServer, mqttPort, 60)
mqtt_client.loop_start()

# -------------------- Configuración de Botón y LED --------------------
# Se utiliza un botón para iniciar el fallo y un LED para indicar visualmente la activación.
from gpiozero import Button, LED
from signal import pause
from flask import Flask, request, jsonify
import threading

# Inicializar la aplicación Flask
app = Flask(__name__)

def run_server():
    # Ejecuta el servidor Flask en todas las interfaces en el puerto 5000
    app.run(host='0.0.0.0', port=5000)

@app.route('/mantto_activo', methods=['POST'])
def mantto_activo():
    # Obtener el JSON enviado en el POST
    data = request.get_json()
    
    # Validar que se reciba un JSON y que contenga el campo 'state'
    if not data or 'state' not in data:
        return jsonify({"error": "Falta el campo 'state' en el JSON"}), 400

    # Extraer el valor del campo 'state'
    state = data['state']

    # Verificar el valor y encender/apagar el LED según corresponda
    if state == "on":
        message = "Encendiendo LED de mantenimiento."
        led_mantto.on()
    elif state == "off": 
        message = "Apagando LED de mantenimiento."
        led_mantto.off()
    else:
        message = "Acción inválida para el LED de mantenimiento. Se espera 'on' o 'off'."
        jsonify({"message": message}), 400
    
    print(message)
    return jsonify({"message": message}), 200

# Configuración del botón y el LEDs 
button = Button(17, pull_up=True)
led_fault = LED(27)
led_mantto = LED(22)

def on_button_pressed():
    """
    Callback del botón: al presionarlo, activa el fallo en las configuraciones y enciende el LED.
    """
    print("Botón presionado... inicio fallo y encendiendo LED.")
    activate_fault([vib_config, curr_config])
    led_fault.on()
    

# Asignar la función de callback al botón
button.when_pressed = on_button_pressed

# -------------------- Uso del Sistema --------------------
if __name__ == "__main__":

    fs = 6              # Frecuencia de muestreo para la simulación (envio muestras por segundo)
    fault_start = 14400   # 4 horas Momento en el que se activa el fallo (en segundos)
    fault_init = fs * 5 # 5 segundos desde oprimir el boton o inicio de falla por tiempo
    duration = 28800 # Duracion total de Simulacion y envio de datos a Thingsboard: 28800 sengudos (8 horas)
    
    # Configuración de la señal de vibración
    vib_config = SignalConfig(
        base=1.0,
        noise=0.02,
        fault_start=fault_start,
        apply_fault=False,
        frequency=50,
        vib_failure_factor=0.5
    )

    # Configuración de la señal de corriente
    curr_config = SignalConfig(
        base=10,
        noise=0.05,
        fault_start=fault_start,
        apply_fault=False,
        frequency=6,         # Este parámetro no se usa en la simulación de corriente, pero se deja para referencia.
        fluctuation=0.05,
        rate=1e-3,
        f_inc=0.35,
        dip_prob=0.005
    )

    server_thread = threading.Thread(target=run_server)
    server_thread.start()

    print("Envío continuo de datos SIN fallo activado.")
    simulate_signals_continuous(vib_config, curr_config, mqtt_client, telemetryTopic, fs, duration, fault_init)
    
    # Mantiene el programa en ejecución para permitir el uso del botón
    pause()
# Finaliza el cliente MQTT y el LED al salir
