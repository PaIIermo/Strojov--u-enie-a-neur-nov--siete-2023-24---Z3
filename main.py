import numpy as np
import pathlib
import os
import gc
import random
import shutil
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from collections import defaultdict
from wordcloud import WordCloud
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

batch_size = 32
img_height = 224
img_width = 224

base_dir = "./z3_data/"
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')  
test_dir = os.path.join(base_dir, 'test')

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    shuffle=True,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
num_classes = len(class_names)

'''
# Class analysis
class_info = {}
for class_name in class_names:
    class_info[class_name] = {'train_samples': 0, 'test_samples': 0,
                              'imagenet_prediction_counts': defaultdict(int),
                              'imagenet_prediction_scores': defaultdict(float)}

unbatched_train_ds = train_ds.unbatch()
unbatched_val_ds = val_ds.unbatch()
unbatched_test_ds = test_ds.unbatch()

saved_images = {}
num_classes = len(class_names)


def process_dataset(dataset, class_info, saved_images, mode):
    for img, label in dataset.as_numpy_iterator():
        class_name = class_names[label]
        class_info[class_name][f'{mode}_samples'] += 1

        if class_name not in saved_images:
            saved_images[class_name] = img.astype("uint8")


process_dataset(unbatched_train_ds, class_info, saved_images, 'train')
process_dataset(unbatched_val_ds, class_info, saved_images, 'train')
process_dataset(unbatched_test_ds, class_info, saved_images, 'test')

del saved_images
del unbatched_train_ds
del unbatched_val_ds
del unbatched_test_ds

gc.collect()
'''

'''
# Displaying example image for each class
images_per_plot = 15
total_images = len(saved_images)
num_plots = (total_images + images_per_plot - 1) 

for plot_index in range(num_plots):
    plt.figure(figsize=(10, 15))

    start_index = plot_index * images_per_plot
    end_index = min(start_index + images_per_plot, total_images)

    for i, (class_name, img) in enumerate(list(saved_images.items())[start_index:end_index]):
        ax = plt.subplot(5, 3, i + 1)
        plt.imshow(img)
        plt.title(class_name)
        plt.axis("off")

    plt.show()
'''


# Initialize the ResNet50 model
resnet_model = ResNet50(weights='imagenet')


# Function to process datasets and aggregate ImageNet prediction data
def process_and_aggregate_data(dataset, class_info):
    for images, labels in dataset:
        predictions = resnet_model.predict(images, verbose=0)
        decoded_predictions = decode_predictions(predictions, top=3)

        for label, top_preds in zip(labels, decoded_predictions):
            class_name = train_ds.class_names[label.numpy()]

            # Update count only for top 1 prediction
            top_pred_class = top_preds[0][1]
            class_info[class_name]['imagenet_prediction_counts'][top_pred_class] += 1

            # Update weighted scores for top 3 predictions
            for _, predicted_class, score in top_preds:
                class_info[class_name]['imagenet_prediction_scores'][predicted_class] += score


def preprocess_for_resnet(image, label):
    return preprocess_input(image), label


preprocessed_train_ds = train_ds.map(preprocess_for_resnet)
preprocessed_val_ds = val_ds.map(preprocess_for_resnet)
preprocessed_test_ds = test_ds.map(preprocess_for_resnet)

# Process each dataset
process_and_aggregate_data(preprocessed_train_ds, class_info)
process_and_aggregate_data(preprocessed_val_ds, class_info)
process_and_aggregate_data(preprocessed_test_ds, class_info)

# Convert cumulative ImageNet scores to average scores
for class_name in class_info:
    total_images = class_info[class_name]['train_samples'] + class_info[class_name]['test_samples']
    for predicted_class in class_info[class_name]['imagenet_prediction_scores']:
        class_info[class_name]['imagenet_prediction_scores'][predicted_class] /= total_images

def generate_latex_table(class_name, info):
    latex_table = f"\\subsection*{{\\textbf{{Class: {class_name}}}}}\n\n"
    latex_table += "\\begin{tabular}{lll}\n"
    latex_table += "\\toprule\n"
    latex_table += "Top Prediction & Count & Score \\\\\n"
    latex_table += "\\midrule\n"

    top_predictions = sorted(info['imagenet_prediction_counts'].items(), key=lambda x: x[1], reverse=True)[:5]
    top_scores = info['imagenet_prediction_scores']

    for i in range(5):
        if i < len(top_predictions):
            pred_class, count = top_predictions[i]
            score = top_scores.get(pred_class, '-')
            if score != '-':
                score = f"{score: .4f}"
            else:
                score = '-'
            pred_class_latex = pred_class.replace('_', '\\_')
            latex_table += f"{pred_class_latex} & {count} & {score} \\\\\n"
        else:
            latex_table += "- & - & - \\\\\n"

    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    return latex_table


# Create a file to save the output
output_file = 'resnet_results.tex'

# Open the file in write mode
with open(output_file, 'w') as file:
    file.write("\\documentclass{article}\n")
    file.write("\\usepackage{multicol}\n")
    file.write("\\usepackage{booktabs}\n")
    file.write("\\begin{document}\n")
    file.write("\\begin{multicols}{2}\n")

    for class_name, info in class_info.items():
        latex_table = generate_latex_table(class_name, info)
        file.write(latex_table)
        file.write("\n")  

    file.write("\\end{multicols}\n")
    file.write("\\end{document}\n")

max_scores_per_class = {}
for class_name, info in class_info.items():
    max_score = max(info['imagenet_prediction_scores'].values(), default=0)
    max_scores_per_class[class_name] = max_score

sorted_classes_by_max_score = sorted(max_scores_per_class.items(), key=lambda x: x[1], reverse=True)
top_3_highest_max_score_classes = sorted_classes_by_max_score[:3]
top_3_lowest_max_score_classes = sorted_classes_by_max_score[-3:]

def plot_top_predictions(class_info, class_name, title):
    top_predictions = sorted(class_info[class_name]['imagenet_prediction_scores'].items(), key=lambda x: x[1],
                             reverse=True)[:3]

    if not top_predictions:
        print(f"No predictions data available for class {class_name}")
        return

    labels, scores = zip(*top_predictions)

    plt.figure(figsize=(6, 4))
    plt.bar(labels, scores, color='skyblue')
    plt.title(f"{class_name} - {title}")
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


for class_name, _ in top_3_highest_max_score_classes:
    plot_top_predictions(class_info, class_name, "Top Predictions (High Confidence)")

for class_name, _ in top_3_lowest_max_score_classes:
    plot_top_predictions(class_info, class_name, "Top Predictions (Low Confidence)")


# Simple way to look for incorrect answers
def is_prediction_correct(resnet_class, actual_class):
    simplified_resnet_class = resnet_class.replace('_', ' ').lower()
    simplified_actual_class = actual_class.replace('_', ' ').lower()
    return simplified_actual_class in simplified_resnet_class

def generate_latex_table_for_incorrect_guesses(incorrect_guesses):
    latex_table = "\\begin{tabular}{ll}\n"
    latex_table += "\\toprule\n"
    latex_table += "Correct Class & Top ResNet Prediction \\\\\n"
    latex_table += "\\midrule\n"

    for class_name, top_resnet_prediction in incorrect_guesses.items():
        class_name_latex = class_name.replace('_', '\\_')
        top_resnet_prediction_latex = top_resnet_prediction.replace('_', '\\_')
        latex_table += f"{class_name_latex} & {top_resnet_prediction_latex} \\\\\n"

    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    return latex_table

incorrect_guesses = {}

for class_name, info in class_info.items():
    top_resnet_prediction = max(info['imagenet_prediction_counts'], key=info['imagenet_prediction_counts'].get, default=None)

    if top_resnet_prediction and not is_prediction_correct(top_resnet_prediction, class_name):
        incorrect_guesses[class_name] = top_resnet_prediction

latex_table = generate_latex_table_for_incorrect_guesses(incorrect_guesses)
output_file = 'incorrect_guesses_table.tex'
with open(output_file, 'w') as file:
    file.write(latex_table)

def create_word_cloud(class_info, class_name):
    wordcloud_data = class_info[class_name]['imagenet_prediction_counts']

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(wordcloud_data)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(class_name)
    plt.axis('off')
    plt.show()


counter = 0

for class_name, info in class_info.items():
    if len(info['imagenet_prediction_scores']) >= 5:
        create_word_cloud(class_info, class_name)
        counter += 1
        if counter >= 4:
            break

# Caching, standardization
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# Model training

# function allowing for runs with multiple models 
def run_experiment(model, id):
    run_dir = f"./model_runs/run_{id}"
    os.makedirs(run_dir, exist_ok=True)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Train the model with EarlyStopping
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[early_stopping]
    )

    history_dict = history.history

    # Plot loss
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history_dict['loss'], label='Training Loss')
    plt.plot(history_dict['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history_dict['accuracy'], label='Training Accuracy')
    plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(os.path.join(run_dir, 'myplot.png'))

    # Confusion matrix
    def get_true_labels_and_predictions(dataset, model):
        y_true = []
        y_pred_classes = []

        for images, labels in dataset:
            y_true.extend(labels.numpy())

            preds = model.predict(images, verbose=0)
            y_pred_classes.extend(np.argmax(preds, axis=1))

        return np.array(y_true), np.array(y_pred_classes)

    y_true_train, y_pred_classes_train = get_true_labels_and_predictions(train_ds, model)

    y_true_test, y_pred_classes_test = get_true_labels_and_predictions(test_ds, model)
    conf_matrix_train = confusion_matrix(y_true_train, y_pred_classes_train)
    conf_matrix_test = confusion_matrix(y_true_test, y_pred_classes_test)

    def plot_confusion_matrix(conf_matrix, title):
        plt.figure(figsize=(12, 12))
        sns.heatmap(conf_matrix, annot=False, fmt='d', cmap='Blues', cbar=False, xticklabels=False, yticklabels=False)
        plt.title(title)
        plt.ylabel('Actual Labels')
        plt.xlabel('Predicted Labels')
        filename = title.replace(' ', '_').replace('/', '_').replace('\\', '_')
        plt.savefig(os.path.join(run_dir, f"{filename}.png"))
        plt.close()

    plot_confusion_matrix(conf_matrix_train, 'Confusion Matrix - Training Data')
    plot_confusion_matrix(conf_matrix_test, 'Confusion Matrix - Testing Data')

    report_train = classification_report(y_true_train, y_pred_classes_train, target_names=class_names)
    report_test = classification_report(y_true_test, y_pred_classes_test, target_names=class_names)

    with open(os.path.join(run_dir, 'classification_report_train.txt'), 'w') as f:
        f.write("Classification Report - Training Data:\n")
        f.write(report_train)

    with open(os.path.join(run_dir, 'classification_report_test.txt'), 'w') as f:
        f.write("Classification Report - Testing Data:\n")
        f.write(report_test)

    train_loss, train_accuracy = model.evaluate(train_ds)
    test_loss, test_accuracy = model.evaluate(test_ds)

    results_df = pd.DataFrame({
        'Dataset': ['Training', 'Testing'],
        'Accuracy': [train_accuracy, test_accuracy]
    })
    results_df.to_csv(os.path.join(run_dir, 'model_accuracy_results.csv'), index=False)


model1 = models.Sequential([
    layers.Rescaling(1. / 255),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(num_classes, activation='softmax')
])

run_experiment(model1, 1)

model2 = models.Sequential([
    layers.Rescaling(1. / 255),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(num_classes, activation='softmax')
])

model3 = models.Sequential([
    layers.Rescaling(1. / 255),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(num_classes, activation='softmax')
])

model4 = models.Sequential([
    layers.Rescaling(1. / 255),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(num_classes, activation='softmax')
])

run_experiment(model2, 2)
run_experiment(model3, 3)
run_experiment(model4, 4)

'''
# Feature extraction 
# Don't run unless you want to wait :)  Outputs a file features.csv. 
# Subsequent runs pull from this file
def get_image_paths_and_labels(directory):
    directory = pathlib.Path(directory)
    image_paths = list(directory.glob('*/*'))
    image_paths = [str(path) for path in image_paths]
    labels = [pathlib.Path(path).parent.name for path in image_paths]
    return image_paths, labels


def extract_features(model, img_batch):
    preprocessed_images = preprocess_input(img_batch)
    features = model(preprocessed_images)
    return features.numpy()


def process_dataset(image_paths, labels, model, dataframe, batch_size=32):
    total_images = len(image_paths)
    for i in range(0, total_images, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        batch_images = []

        for img_path in batch_paths:
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            batch_images.append(img_array)

        img_batch = np.stack(batch_images, axis=0)
        features_batch = extract_features(model, img_batch)

        for features, img_path, label in zip(features_batch, batch_paths, batch_labels):
            label_index = class_names.index(label)
            row = [img_path] + list(features) + [label_index]
            dataframe.loc[len(dataframe)] = row

        print(f"Processed {min(i + batch_size, total_images)} / {total_images} images")


# Initialize ResNet50 model for feature extraction
base_model = ResNet50(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet', pooling='avg')
base_model.trainable = False

# Initialize dataframe to store features and labels
num_features = base_model.output_shape[1]
columns = ['image_path'] + [f'feature_{i}' for i in range(num_features)] + ['label']
df_features = pd.DataFrame(columns=columns)

# Process each dataset
train_image_paths, train_labels = get_image_paths_and_labels(train_dir)
val_image_paths, val_labels = get_image_paths_and_labels(val_dir)
test_image_paths, test_labels = get_image_paths_and_labels(test_dir)

process_dataset(train_image_paths, train_labels, base_model, df_features, batch_size=32)
process_dataset(val_image_paths, val_labels, base_model, df_features, batch_size=32)
process_dataset(test_image_paths, test_labels, base_model, df_features, batch_size=32)

df_features.to_csv('features.csv', index=False)
'''


# Dimension reduction using PCA
df_features = pd.read_csv('features.csv')
features = df_features.iloc[:, 1:-1]

pca = PCA(n_components=50)  
reduced_features = pca.fit_transform(features)

# Perform K-Means clustering on reduced features, 
# expecting a predefined number of clusters
num_clusters = 10  
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(reduced_features)

df_features['cluster'] = kmeans.labels_


'''
# Visualization of example collage and centroid image for each cluster
def save_image_collage_from_cluster(df, base_dir, cluster_label, num_images=5):
    cluster_images = df[df['cluster'] == cluster_label]['image_path'].tolist()
    selected_images = cluster_images[:min(num_images, len(cluster_images))]

    # Load images and resize them
    images = [Image.open(img_path).resize((100, 100)) for img_path in selected_images]

    # Create a collage
    collage_width = 100 * len(images)
    collage = Image.new('RGB', (collage_width, 100))
    for i, img in enumerate(images):
        collage.paste(img, (i * 100, 0))

    # Save the collage in the base directory
    collage_path = os.path.join(base_dir, f'cluster_{cluster_label}_collage.jpg')
    collage.save(collage_path)

base_collage_dir = 'cluster_collages'  # Base directory for all collages
os.makedirs(base_collage_dir, exist_ok=True)  # Create base directory

for cluster_label in range(num_clusters):
    save_image_collage_from_cluster(df_features, base_collage_dir, cluster_label, num_images=5)
def save_representative_images(df, kmeans_model, pca_model, base_dir, num_clusters):
    for cluster_label in range(num_clusters):
        cluster_df = df[df['cluster'] == cluster_label]

        # Applying PCA transformation to the features of the current cluster
        cluster_features_reduced = pca_model.transform(cluster_df.iloc[:, 1:-2])

        # Calculating the distance to the centroid
        centroid = kmeans_model.cluster_centers_[cluster_label]
        closest_img_idx = np.argmin(np.linalg.norm(cluster_features_reduced - centroid, axis=1))
        closest_img_path = cluster_df.iloc[closest_img_idx]['image_path']

        # Create a directory for the representative image within the base directory
        rep_img_dir = os.path.join(base_dir, f'cluster_{cluster_label}_representative')
        os.makedirs(rep_img_dir, exist_ok=True)

        shutil.copy(closest_img_path, os.path.join(rep_img_dir, os.path.basename(closest_img_path)))

save_representative_images(df_features, kmeans, pca, base_collage_dir, num_clusters=num_clusters)
'''

# Random forest training:
X = df_features.iloc[:, 1:-2]
y = df_features['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)

clf.fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Calculate accuracies
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print accuracies
print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

report = classification_report(y_test, y_test_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
report_file = "classification_report.csv"
df_report.to_csv(report_file)

cm1 = confusion_matrix(y_train, y_train_pred)
cm2 = confusion_matrix(y_test, y_test_pred)


def plot_confusion_matrix(conf_matrix, title):
    plt.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, annot=False, fmt='d', cmap='Blues', cbar=False, xticklabels=False, yticklabels=False)
    plt.title(title)
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')
    plt.show()


plot_confusion_matrix(cm1, 'Confusion Matrix - Train Data')
plot_confusion_matrix(cm2, 'Confusion Matrix - Test Data')

