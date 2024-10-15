import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from yamnet import params as yamnet_params
from yamnet import yamnet as yamnet_model

class Clasificador:
    def __init__(self, units=100, alpha=0.001):
        #self.yamnet_modelo = self.load_yamnet_model()
        #self.input_shape = self.yamnet_modelo.input.shape[1:]
        self.input_shape = (1024,)
        self.units = units
        self.alpha = alpha
        self.model = self._build_model()

    def load_yamnet_model(self):
        params = yamnet_params.Params(patch_hop_seconds=0.96)
        yamnet = yamnet_model.yamnet_frames_model(params)
        yamnet.load_weights('C:\\Users\\user\\Documents\\Tesis\\codigo\\yamnet\\yamnet.h5')
        yamnet.traineable = False
        return yamnet
    
    def _build_model(self):
        input_layer = tf.keras.layers.Input(self.input_shape, dtype=tf.float32)
        second_layer = tf.keras.layers.Dense(self.units, activation='relu')(input_layer) #(yamnet_output)
        output_layer = tf.keras.layers.Dense(1, 'sigmoid')(input_layer) #(second_layer)
        clasificador = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        return clasificador
    
    def compilar(self):
        self.model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="Accuracy"),
                tf.keras.metrics.Precision(name='Precision', thresholds=self._thresholds()),
                tf.keras.metrics.Recall(name='Recall', thresholds=self._thresholds()),
                tf.keras.metrics.TruePositives(name='TP', thresholds=self._thresholds()),
                tf.keras.metrics.TrueNegatives(name='TN', thresholds=self._thresholds()),
                tf.keras.metrics.FalsePositives(name='FP', thresholds=self._thresholds()),
                tf.keras.metrics.FalseNegatives(name='FN', thresholds=self._thresholds())
            ]
        )

    def _thresholds(self):
        return np.linspace(0.01, 1, 100, dtype=np.float64).round(2).tolist()

    def train(self, train_ds, val_ds, epochs=60):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                          patience=20, 
                                                          restore_best_weights=True)
        history = self.model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=early_stopping)
        return history
    
    def plot_training_curves(self, history):
        train_acc = history.history['Accuracy']
        val_acc = history.history['val_Accuracy']
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']

        i = np.linspace(0.01, 1, 100, dtype=np.float64).round(2).tolist().index(0.5)
        train_rec = np.array(history.history['Recall'])[:,i]
        val_rec = np.array(history.history['val_Recall'])[:,i]
        train_pre = np.array(history.history['Precision'])[:,i]
        val_pre = np.array(history.history['val_Precision'])[:,i]

        epochs = range(1, len(train_acc) + 1)
        
        fig, ax = plt.subplots(2, 2, figsize=(14, 10))
        ax[0,0].plot(epochs, train_acc, 'bo', label='Exactitud de Entrenamiento')
        ax[0,0].plot(epochs, val_acc, 'b', label='Exactitud de Validacion')
        ax[0,0].set_title('Exactitud (Accuracy)')
        ax[0,0].legend()
        ax[0,0].grid(True)
        ax[0,0].set_ylim([0.8, 1.01])

        ax[0,1].plot(epochs, train_loss, 'bo', label='Perdida de Entrenamiento')
        ax[0,1].plot(epochs, val_loss, 'b', label='Perdida de validacion')
        ax[0,1].set_title('Perdida (Loss)')
        ax[0,1].legend()
        ax[0,1].grid(True)
        ax[0,1].set_ylim([0.0, 1.1])

        ax[1,0].plot(epochs, train_rec, 'bo', label='Recall de Entrenamiento')
        ax[1,0].plot(epochs, val_rec, 'b', label='Recall de Validacion')
        ax[1,0].set_title('Exaustividad (Recall)')
        ax[1,0].legend()
        ax[1,0].grid(True)
        ax[1,0].set_ylim([0.8, 1.01])

        ax[1,1].plot(epochs, train_pre, 'bo', label='Precision de Entrenamiento')
        ax[1,1].plot(epochs, val_pre, 'b', label='Precision de validacion')
        ax[1,1].set_title('Precision')
        ax[1,1].legend()
        ax[1,1].grid(True)
        ax[1,1].set_ylim([0.8, 1.01])
        
        plt.tight_layout()
        plt.show()
    
    def exportar_entrenamiento(self, history, name):
        import pickle
        with open(name, 'wb') as file:
            pickle.dump(history.history, file)
        return None

    

if __name__ == "__main__":
    # Example usage:
    from dataLoader import AudioDataset
    metadata_path = "C:\\Users\\user\\Documents\\Tesis\\metadata.csv"
    audio_dir = "C:\\Users\\user\\Documents\\Tesis\\dataset\\audios"

    audio_dataset = AudioDataset(metadata_path, audio_dir)
    train_ds, val_ds, test_ds = audio_dataset.preprocess_datasets()

    classifier = Clasificador()
    classifier.compilar()
    history = classifier.train(train_ds, val_ds)

