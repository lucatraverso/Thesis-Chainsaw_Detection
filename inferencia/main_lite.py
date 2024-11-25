import sounddevice as sd
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
import soundfile as sf

# Define constants for audio recording
SAMPLE_RATE = 16000  # Sample rate (Hz)
CHANNELS = 1         # Number of audio channels
I = 1
BLOCK_LENGTH = int(I*0.96*SAMPLE_RATE)

try:
    model_path = 'C:\\Users\\user\\Documents\\Tesis\\codigo\\modelo_final.tflite'
    interpreter = tf.lite.Interpreter(model_path)

    input_details = interpreter.get_input_details()
    waveform_input_index = input_details[0]['index']
    output_details = interpreter.get_output_details()
    scores_output_index = output_details[0]['index']

    interpreter.resize_tensor_input(waveform_input_index, [BLOCK_LENGTH], strict=True)
    interpreter.allocate_tensors()
except:
    print('Error cargando el modelo')

recorded_audio = []
dic = {'Inicio':[], 'Fin':[]}
# Function to process audio data
def process_audio(indata, frames, time, status):
    # Convert audio data to numpy array
    global recorded_audio
    audio_data = np.asarray(indata, np.float32)
    audio_data = audio_data.reshape(-1)

    ti = timedelta(seconds=(len(recorded_audio) / 16000))
    recorded_audio.extend(audio_data)
    tf = timedelta(seconds=  (len(recorded_audio) / 16000))

    interpreter.set_tensor(waveform_input_index, audio_data)
    interpreter.invoke()
    output = interpreter.get_tensor(scores_output_index)
    alerta = np.all(output > 0.88)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if alerta:
        print(f'Motosierra! {timestamp}')
        dic['Inicio'].append(ti)
        dic['Fin'].append(tf)
    else:
        print('Todo ok')
    

# Open a stream for audio input
stream = sd.InputStream(callback=process_audio, blocksize=BLOCK_LENGTH, 
                        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.float32)

print("Recording... Press Ctrl+C to stop.")

# Start audio stream
with stream:
    try:
        # Keep the stream open until interrupted by Ctrl+C
        while True:
            pass
    except KeyboardInterrupt:
        pass

print("Finished recording.")


recorded_audio = np.array(recorded_audio)
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filename = f'{timestamp}.wav'
sf.write(filename, recorded_audio, SAMPLE_RATE)

import pandas as pd
pd.DataFrame(dic).to_csv(f'{timestamp}_log.csv', index=False)

print(f'Audio saved as {filename}')