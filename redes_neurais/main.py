import os
import cv2
from PIL import Image
import numpy as np
from keras import layers
from keras.optimizers import Nadam
from keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from keras.applications import ResNet50, EfficientNetB4
from keras.applications.resnet import ResNet101
from keras.wrappers.scikit_learn import KerasClassifier
from datetime import datetime


class MorphologicalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, kernel_size=5, kernel_shape=cv2.MORPH_CROSS, operation_type='erosao', iterations=1):
        self.kernel_size = kernel_size
        self.kernel_shape = kernel_shape
        self.iterations = iterations
        self.operation_type = operation_type

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_images = []
        kernel = cv2.getStructuringElement(self.kernel_shape, (self.kernel_size, self.kernel_size))

        for image in X:
            if self.operation_type == 'erosao':
                transformed_image = cv2.erode(image, kernel, iterations=self.iterations)
            elif self.operation_type == 'dilatacao':
                transformed_image = cv2.dilate(image, kernel, iterations=self.iterations)
            elif self.operation_type == 'abertura':
                transformed_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=self.iterations)
            elif self.operation_type == 'fechamento':
                transformed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=self.iterations)
            else:
                raise ValueError('Invalid operation type.')

            transformed_images.append(transformed_image)

        return np.array(transformed_images)

def create_model(nome_modelo, tarefa_classificacao):
    if nome_modelo == 'EfficientNetB4': base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(224,224,3))
    elif nome_modelo == 'ResNet50': base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
    elif nome_modelo == 'ResNet101': base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(224,224,3))
    else: raise ValueError('Invalid model.')
    
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)
    n_classes, activation = (8, "softmax") if tarefa_classificacao == 'classificacao_multiclasse' else (1, "sigmoid")
    predictions = layers.Dense(n_classes, activation=activation)(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    loss = "categorical_crossentropy" if tarefa_classificacao == 'classificacao_multiclasse' else "binary_crossentropy"

    model.compile(optimizer=Nadam(1e-4),
                loss=loss,  
                metrics=["accuracy"])

    return model

def load_images(root_dir):
    images = []
    labels = []
    label_dict = {}

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
                img = Image.open(os.path.join(subdir, file))
                img_arr = np.array(img)
                images.append(img_arr)
                label = subdir.split("/")[-1]
                if label not in label_dict:
                    label_dict[label] = len(label_dict)
                numeric_label = label_dict[label]
                labels.append(numeric_label)

    return np.array(images), np.array(labels)

metricas_b = ['accuracy', 'precision', 'recall', 'f1']
metricas_m = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
tarefas_classificacao = ['classificacao_binaria','classificacao_multiclasse']
ampliacoes = ['40X', '100X', '200X', '400X']
nomes_modelos = ['EfficientNetB4', 'ResNet50', 'ResNet101']
pre_processamentos = ['original', 'dilatacao', 'abertura', 'fechamento', 'erosao']
EPOCHS = 10
BATCH_SIZE = 8
VERBOSE = 1
CV_KFOLD = 5

arq = f"CNN_{datetime.now().strftime('%d-%m-%Y %H-%M-%S')}.txt"
with open(arq, 'w') as f:
    f.write(f'TCC Mayke: CNN - EPOCHS {EPOCHS} - BATCH_SIZE {BATCH_SIZE} - CV_KFOLD {CV_KFOLD}\n')

for tarefa_classificacao in tarefas_classificacao:
    for ampliacao in ampliacoes:
        root_dir = f'dataset_cancer_v1/{tarefa_classificacao}/{ampliacao}'
        images, labels = load_images(root_dir)
        for nome_modelo in nomes_modelos:
            for pre_processamento in pre_processamentos:

                pipeline = []

                if pre_processamento != 'original': 
                    pipeline.append(('morfologia', MorphologicalTransformer(5, cv2.MORPH_RECT, pre_processamento)))

                pipeline.extend([
                    ('classify', KerasClassifier(build_fn=create_model, nome_modelo=nome_modelo, tarefa_classificacao=tarefa_classificacao, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE))
                ])

                pipeline = Pipeline(pipeline)

                cv_stratified = StratifiedKFold(CV_KFOLD, shuffle=True, random_state=42)

                for train, test in cv_stratified.split(images, labels):
                    X_train = images[train, :]
                    y_train = labels[train]
                    X_test = images[test, :]
                    y_test = labels[test]

                    metricas = metricas_m if tarefa_classificacao == 'classificacao_multiclasse' else metricas_b
                    
                    cv_results = cross_validate(pipeline, X_train, y_train, scoring=metricas, cv=cv_stratified, return_estimator=True)
                    
                    best_estimator_index = cv_results['test_accuracy'].argmax()
                    best_estimator = cv_results['estimator'][best_estimator_index]

                    y_pred = best_estimator.predict(X_test)

                    accuracy = accuracy_score(y_test, y_pred)
                    if tarefa_classificacao == 'classificacao_multiclasse':
                        precision = precision_score(y_test, y_pred, average='weighted')
                        recall = recall_score(y_test, y_pred, average='weighted')
                        f1 = f1_score(y_test, y_pred, average='weighted')
                    else:
                        precision = precision_score(y_test, y_pred)
                        recall = recall_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred)
                    
                    with open(arq, 'a') as f:
                        f.write(f'\n{nome_modelo}\t{tarefa_classificacao}\t{ampliacao}\t{pre_processamento}')
                        for i in cv_results:
                            if i != 'estimator': 
                                f.write("\t{:.4f}".format(cv_results[i].mean()).replace(".", ","))
                                f.write("\t{:.4f}".format(cv_results[i].std()).replace(".", ","))
                        f.write("\t{:.4f}".format(accuracy).replace(".", ","))
                        f.write("\t{:.4f}".format(precision).replace(".", ","))
                        f.write("\t{:.4f}".format(recall).replace(".", ","))
                        f.write("\t{:.4f}".format(f1).replace(".", ","))