import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# np.random.seed(42)
# random.seed(42)
# tf.random.set_seed(42)

# (X_train, y_train), (X_test, y_test)= keras.datasets.mnist.load_data()
# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)

# X_train= X_train.astype("float32") / 255.0
# X_test= X_test.astype("float32") / 255.0
# X_train= np.expand_dims(X_train, -1)
# X_test= np.expand_dims(X_test, -1)
# print("After preprocessing:")
# print(X_train.shape, X_test.shape)

# plt.figure(figsize=(8, 8))
# for i in range(9):
#     idx = random.randint(0, X_train.shape[0] - 1)
#     plt.subplot(3, 3, i + 1)
#     plt.imshow(X_train[idx].reshape(28, 28), cmap="gray")
#     plt.title(f"Label: {y_train[idx]}")
#     plt.axis("off")
# plt.show()

# sns.countplot(x=y_train, palette="viridis")
# plt.title("Digit distribution in training set")
# plt.show()

# X_train_full, X_val, y_train_full, y_val = train_test_split(
#     X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
# )

# model= keras.Sequential([
#     layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation="relu"),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(128, activation="relu"),
#     layers.Dropout(0.5),
#     layers.Dense(10, activation="softmax")
# ])

# model.summary()
# model.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# datagen= keras.preprocessing.image.ImageDataGenerator(
#     rotation_range=10,       
#     width_shift_range=0.1,  
#     height_shift_range=0.1,  
#     zoom_range=0.1           
# )

# my_callbacks = [
#     EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
#     ModelCheckpoint("best_mnist_model.h5", save_best_only=True, monitor="val_loss")
# ]

# history= model.fit(
#     datagen.flow(X_train_full, y_train_full, batch_size=64),
#     epochs=10,  
#     validation_data=(X_val, y_val),
#     callbacks=my_callbacks,
#     verbose=2
# )

# best_model= keras.models.load_model("best_mnist_model.h5")
# best_loss, best_acc= best_model.evaluate(X_test, y_test, verbose=0)
# print(f"Best Model - Accuracy: {best_acc:.4f}, Loss: {best_loss:.4f}")

# y_pred= np.argmax(best_model.predict(X_test), axis=1)
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.show()

# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# plt.figure(figsize=(8, 4))
# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Val Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Model Accuracy')
# plt.legend()
# plt.show()

# plt.figure(figsize=(8, 4))
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Val Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Model Loss')
# plt.legend()
# plt.show()

# X_train_flat= X_train_full.reshape(len(X_train_full), -1)
# X_val_flat= X_val.reshape(len(X_val), -1)
# baseline_model= LogisticRegression(max_iter=1000)
# baseline_model.fit(X_train_flat, y_train_full)
# y_val_pred= baseline_model.predict(X_val_flat)
# baseline_acc= accuracy_score(y_val, y_val_pred)
# print("Baseline Logistic Regression Accuracy:", baseline_acc)

# rf_model= RandomForestClassifier(n_estimators=100, random_state=42)
# rf_model.fit(X_train_flat, y_train_full)
# y_val_pred_rf= rf_model.predict(X_val_flat)
# rf_acc= accuracy_score(y_val, y_val_pred_rf)
# print("Random Forest Accuracy:", rf_acc)

# dt_model= DecisionTreeClassifier(max_depth=10, random_state=42)
# dt_model.fit(X_train_flat, y_train_full)
# y_val_pred_dt= dt_model.predict(X_val_flat)
# dt_acc= accuracy_score(y_val, y_val_pred_dt)
# print("Decision Tree Accuracy:", dt_acc)

# results= pd.DataFrame({
#     "Model": ["Logistic Regression", "Random Forest", "Decision Tree", "CNN"],
#     "Validation Accuracy": [baseline_acc, rf_acc, dt_acc, best_acc]
# })
# results= results.sort_values(by="Validation Accuracy", ascending=False)
# print(results)

# best_model_name= results.iloc[0]['Model']
# print('Best Model:', best_model_name)

# if best_model_name=="CNN":
#     X_eval= X_test
#     y_pred_final= np.argmax(best_model.predict(X_eval), axis=1)
# else:
#     X_eval= X_test.reshape(len(X_test), -1)
#     if best_model_name== "Logistic Regression":
#         y_pred_final= baseline_model.predict(X_eval)
#     elif best_model_name== "Random Forest":
#         y_pred_final= rf_model.predict(X_eval)
#     else:
#         y_pred_final= dt_model.predict(X_eval)

# plt.figure(figsize=(8, 8))
# for i in range(9):
#     idx = random.randint(0, len(X_test) - 1)
#     plt.subplot(3, 3, i + 1)
#     plt.imshow(X_test[idx].reshape(28, 28), cmap="gray")
#     plt.title(f"True: {y_test[idx]}, Pred: {y_pred_final[idx]}")
#     plt.axis("off")
# plt.show()

# if best_model_name== "CNN":
#     best_model.save("final_best_model.h5")
# else:
#     import joblib
#     if best_model_name== "Logistic Regression":
#         joblib.dump(baseline_model, "final_best_model.pkl")
#     elif best_model_name== "Random Forest":
#         joblib.dump(rf_model, "final_best_model.pkl")
#     else:
#         joblib.dump(dt_model, "final_best_model.pkl")
# print("Best model saved successfully.")

# plt.figure(figsize=(6, 4))
# sns.barplot(data=results, x="Validation Accuracy", y="Model", palette="viridis")
# plt.xlim(0, 1)
# plt.title("Model Validation Accuracy Comparison")
# plt.show()
