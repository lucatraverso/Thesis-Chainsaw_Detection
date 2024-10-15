import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class ClassifierEvaluator:
    def __init__(self, model):
        self.model = model
        self.umbral = 0.5
        self.indx = self._thresholds().index(self.umbral)
    
    def _thresholds(self):
        return np.linspace(0.01, 1, 100, dtype=np.float64).round(2).tolist()
    

    def plot_training_curves(self, history):
        train_acc = history.history['Accuracy']
        val_acc = history.history['val_Accuracy']
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(train_acc) + 1)
        
        fig, ax = plt.subplots(2, 2, figsize=(14, 10))
        
        ax[0,0].plot(epochs, train_acc, 'bo', label='Training Accuracy')
        ax[0,0].plot(epochs, val_acc, 'b', label='Validation Accuracy')
        ax[0,0].set_title('Accuracy')
        ax[0,0].legend()
        ax[0,0].grid(True)
        
        ax[0,1].plot(epochs, train_loss, 'bo', label='Training Loss')
        ax[0,1].plot(epochs, val_loss, 'b', label='Validation Loss')
        ax[0,1].set_title('Loss')
        ax[0,1].legend()
        ax[0,1].grid(True)
        
        ax[1,0].set_visible(False)
        ax[1,1].set_visible(False)
        
        plt.tight_layout()
        plt.show()

    def evaluate(self, test_ds):
        loss, bacc, precision, recall, TP, TN, FP, FN = self.model.evaluate(test_ds)
        F1 = (2 * precision * recall) / (precision + recall)

        print(f'Loss: {loss:.2f}')
        print(f'Accuracy: {bacc:.2f}')
        print(f'Precision: {precision[self.indx]:.2f}')
        print(f'Recall: {recall[self.indx]:.2f}')
        print(f'F1: {F1[self.indx]:.2f}')
        print('TP: ', TP[self.indx])
        print('TN: ', TN[self.indx])
        print('FP: ', FP[self.indx])
        print('FN: ', FN[self.indx])

        return loss, bacc, precision, recall, F1, TP, TN, FP, FN
    
    def plot_confusion_matrix(self, TP, TN, FP, FN):
        pos = TP[0] + FN[0]
        neg = TN[0] + FP[0]
        
        fig, ax = plt.subplots()
        sns.heatmap([[TP[self.indx]/pos, FN[self.indx]/pos], [FP[self.indx]/neg, TN[self.indx]/neg]], 
                    xticklabels=['Predicted Positive', 'Predicted Negative'], 
                    yticklabels=['True Positive', 'True Negative'], 
                    annot=[[TP[self.indx]/pos, FN[self.indx]/pos], [FP[self.indx]/neg, TN[self.indx]/neg]],
                    fmt='.2f',
                    cmap='Blues',
                    ax=ax)
        ax.set_xlabel('Predicted Value')
        ax.set_ylabel('True Value')
        ax.set_title('Confusion Matrix')
        plt.show()

    def curva_det(self, TP, TN, FP, FN):
        segmentos = TP[0] + FN[0] + TN[0] + FP[0]
        tiempo = (segmentos * 0.96) / 3600
        FPH = FP / tiempo
        TNR = FN / (TP + FN)
        th = np.linspace(0.01, 1, 100, dtype=np.float64).round(2).tolist()
        start = th.index(0.01)
        end = th.index(0.99)

        plt.plot(FPH[start:end], TNR[start:end])

        # Add markers or annotations for the values in th
        for i, threshold in enumerate(th[start:end]):
            if i % 10 == 0:
                plt.text(FPH[i], TNR[i], str(threshold), fontsize=8, ha='right', va='bottom')

        plt.xlabel('False Positives per Hour')
        plt.ylabel('False Negative Rate')
        plt.title('DET Curve with Threshold Markers')
        plt.grid(True)
        plt.show()
    