from utils import Utils
from sklearn.model_selection import train_test_split
import tensorflow as tf


def train_and_save_model(images, labels):
    images, labels = Utils.load_and_preprocess_data(
        "training_data", "labels.csv", 400, 400
    )

    train_x, test_x, train_y, test_y = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )

    train_hypertune_x, val_x, train_hypertune_y, val_y = train_test_split(
        train_x, train_y, test_size=0.25, random_state=42
    )

    # Hyperparameter tuning
    optimizer_list = ["Adam", "Adagrad", "Nadam"]
    lr_list = [0.001, 0.01, 0.1]
    batch_list = [32, 64, 128]
    best_loss, best_params = Utils.hyperparameter_tuning(
        optimizer_list,
        lr_list,
        batch_list,
        train_hypertune_x,
        val_x,
        train_hypertune_y,
        val_y,
    )

    # Architecture tuning
    conv_layers_list = [[32, 64], [32, 64, 128], [64, 128, 256]]
    filter_dim_list = [(3, 3), (5, 5)]
    pooling_dim_list = [(2, 2), (3, 3)]
    fc_layers_list = [[512], [512, 256]]
    dropout_val_list = [0.3, 0.5]

    best_architecture = Utils.architecture_tuning(
        best_params,
        conv_layers_list,
        filter_dim_list,
        pooling_dim_list,
        fc_layers_list,
        dropout_val_list,
        train_x,
        test_x,
        train_y,
        val_y,
    )

    print(f"Best Hyperparameters: {best_params}")
    print(f"Best Architecture: {best_architecture}")

    # Build and train the final model with the best parameters and architecture
    model_spec = {
        "conv_layers": best_architecture["conv_layers"],
        "filter_dim": best_architecture["filter_dim"],
        "pooling_dim": best_architecture["pooling_dim"],
        "fc_layers": best_architecture["fc_layers"],
        "dropout_val": best_architecture["dropout_val"],
        "input_shape": (train_x.shape[1], train_x.shape[2], 3),
        "optimizer": best_params["optimizer"],
        "learning_rate": best_params["learning_rate"],
        "loss": "binary_crossentropy",
    }

    model = Utils.build_model(model_spec)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    model.fit(
        train_x,
        train_y,
        epochs=100,
        batch_size=best_params["batch_size"],
        validation_data=(test_x, test_y),
        callbacks=[early_stopping],
        verbose=1,
    )

    # Save the model
    model.save("best_model.h5")
