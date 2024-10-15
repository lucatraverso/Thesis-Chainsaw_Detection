import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow_io as tfio
from yamnet import params as yamnet_params
from yamnet import yamnet as yamnet_model

class AudioDataset:
    def __init__(self, metadata_path, audio_dir, split=True, BATCH_SIZE=32):
        self.metadata_path = metadata_path
        self.audio_dir = audio_dir
        self.BATCH_SIZE = BATCH_SIZE
        self.dataset = self._load_dataset()
        self.yamnet = self._load_yamnet_model()
        self._create_filepaths()
        self._convert_to_numpy()
        if split == True:
            self._split_dataset()
            self._create_tf_dataset()
            self._map_audios()
            

    def _load_dataset(self):
        dataset = pd.read_csv(self.metadata_path)
        return dataset[['filename', 'category']]

    def _create_filepaths(self):
        self.dataset['filename'] = self.dataset['filename'].apply(lambda fname: os.path.join(self.audio_dir, fname))

    def _convert_to_numpy(self):
        self.filename = self.dataset['filename'].to_numpy()
        self.category = np.array(self.dataset['category'].to_list())

    def _split_dataset(self):
        self.x_train, self.x_temp, self.y_train, self.y_temp = train_test_split(self.filename, self.category, test_size=0.3, random_state=42, shuffle=True)
        self.x_val, self.x_test, self.y_val, self.y_test = train_test_split(self.x_temp, self.y_temp, test_size=0.5, random_state=42, shuffle=True)
    
    def _create_tf_dataset(self):
        self.train_ds = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        self.val_ds = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val))
        self.test_ds = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
    
    def create_single_dataset(self):
        self.ds = tf.data.Dataset.from_tensor_slices((self.filename, self.category))
        self.ds = self.ds.map(self._extract_embedding).unbatch()
        self.ds = self.ds.cache().shuffle(self.filename.shape[0]).batch(self.BATCH_SIZE)
        return self.ds

    def _load_yamnet_model(self):
        params = yamnet_params.Params(patch_hop_seconds=0.96)
        yamnet = yamnet_model.yamnet_frames_model(params)
        yamnet.load_weights('C:\\Users\\user\\Documents\\Tesis\\codigo\\yamnet\\yamnet.h5')
        yamnet.traineable = False
        return yamnet

    def _load_wav_16k_mono(self, fpath):
        file_contents = tf.io.read_file(fpath)
        wav, sample_rate = tf.audio.decode_wav(
            file_contents,
            desired_channels=1)
        wav = tf.squeeze(wav, axis=-1)
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
        return wav

    def _extract_embedding(self, path, label):
        wav_data = self._load_wav_16k_mono(path)
        scores, embeddings, spectrogram = self.yamnet(wav_data)
        num_embeddings = tf.shape(embeddings)[0]
        return (embeddings, tf.repeat([label], num_embeddings, axis=0))

    def _map_audios(self):
        self.train_ds = self.train_ds.map(self._extract_embedding).unbatch()
        self.val_ds = self.val_ds.map(self._extract_embedding).unbatch()
        self.test_ds = self.test_ds.map(self._extract_embedding).unbatch()

    def preprocess_datasets(self):
        self.train_ds = self.train_ds.cache().shuffle(self.x_train.shape[0]).batch(self.BATCH_SIZE)
        self.val_ds = self.val_ds.cache().batch(self.BATCH_SIZE)
        self.test_ds = self.test_ds.cache().batch(self.BATCH_SIZE)
        return self.train_ds, self.val_ds, self.test_ds
    
if __name__ == "__main__":
    # Usage
    metadata_path = ".\\metadata.csv"
    audio_dir = ".\\dataset\\audios"


    audio_dataset = AudioDataset(metadata_path, audio_dir)
    train_ds, val_ds, test_ds = audio_dataset.preprocess_datasets()

