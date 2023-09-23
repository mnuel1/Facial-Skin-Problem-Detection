import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import random
import tensorflow as tf

class SkinDiseaseEvaluator:
    def __init__(self):
        # Set random seeds for reproducibility
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)

        # Configuration
        self.BATCH_SIZE = 16
        
        self.TEST_PATH = 'datasets/archive_3/Original Images/Original Images/FOLDS/fold1/Test'
        

    def load_model_and_evaluate(self):
        # Load test data
        test_data = self.load_test_data()

        # Load pre-trained model
        model = load_model('my_model.h5')

        # Predict test data
        model_preds = model.predict(test_data, test_data.samples // test_data.batch_size + 1)
        model_pred_classes = np.argmax(model_preds, axis=1)

        # self.plot_training_history(model.history)
        # Evaluate the model
        self.evaluate_model(test_data, model_pred_classes)

    def load_test_data(self):
        tsdata = ImageDataGenerator()
        return tsdata.flow_from_directory(
            directory=self.TEST_PATH,
            target_size=(224, 224),
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            seed=42
        )

    def evaluate_model(self, test_data, model_pred_classes):
        true_classes = test_data.classes

        # Calculate accuracy, precision, recall, and F1-score
        acc = accuracy_score(true_classes, model_pred_classes)
        precision = precision_score(true_classes, model_pred_classes, average='macro')
        recall = recall_score(true_classes, model_pred_classes, average='macro')
        f1 = f1_score(true_classes, model_pred_classes, average='macro')

        # Print evaluation metrics
        print("DenseNet121-based Model Accuracy: {:.2f}%".format(acc * 100))
        print('Precision: %.3f' % precision)
        print('Recall: %.3f' % recall)
        print('F1 Score: %.3f' % f1)

        # Generate a classification report
        print('Classification Report')
        target_names = ['Chickenpox', 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Monkeypox']
        print(classification_report(true_classes, model_pred_classes))

        # Generate a confusion matrix and plot it
        x = confusion_matrix(true_classes, model_pred_classes)
        plot_confusion_matrix(x)

        # Plot heatmap
        class_names = test_data.class_indices.keys()
        fig, (ax1) = plt.subplots(1, 1, figsize=(10, 10))
        self.plot_heatmap(true_classes, model_pred_classes, class_names, ax1, title="DenseNet121")    
        fig.suptitle("Confusion Matrix Model Comparison", fontsize=24)
        fig.tight_layout()
        
        plt.show()

        
        
    def plot_heatmap(self, y_true, y_pred, class_names, ax, title):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True,
                xticklabels=class_names, yticklabels=class_names)
        ax.set_title(title, fontsize=16)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)



if __name__ == "__main__":
    evaluator = SkinDiseaseEvaluator()
    evaluator.load_model_and_evaluate()