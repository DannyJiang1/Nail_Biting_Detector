import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.api.preprocessing.image import img_to_array, load_img
from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.api.optimizers import Adam, Adagrad, Nadam


class Utils:
    def load_and_preprocess_data(
        self, data_folder, csv_file, img_height, img_width
    ):
        labels_df = pd.read_csv(csv_file)
        images = []
        labels = []

        for row in labels_df.itertuples(index=False):
            filepath = os.path.join(data_folder, row.filename)
            if os.path.exists(filepath):
                img = load_img(filepath, target_size=(img_height, img_width))
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(row.label)

        images = np.array(images, dtype="float32") / 255.0
        labels = np.array(labels)

        return images, labels

    def build_model(self, model_spec):
        """model_spec:
        {
        conv_layers:
            [
                filters: int
            ],
        filter_dim: tuple,
        pooling_dim: tuple,
        fc_layers:
            [
                units: int
            ],
        dropout_val: float,
        input_shape: tuple,
        optimizer: string,
        learning_rate: float,
        loss: string
        custom_loss : loss_function (if loss == "custom")
        }
        """
        # Build the model
        model = Sequential()
        for i, layer_filters in enumerate(model_spec["conv_layers"]):
            if i == 0:
                model.add(
                    Conv2D(
                        filters=layer_filters,
                        kernel_size=model_spec["filter_dim"],
                        input_shape=model_spec["input_shape"],
                        activation="relu",
                    )
                )
            else:
                model.add(
                    Conv2D(
                        filters=layer_filters,
                        kernel_size=model_spec["filter_dim"],
                        activation="relu",
                    )
                )
            model.add(MaxPooling2D(model_spec["pooling_dim"]))

        model.add(Flatten)

        for layer_units in model_spec["fc_layers"]:
            model.add(Dense(units=layer_units, activation="relu"))
            model.add(Dropout(model_spec["dropout_val"]))

        model.add(Dense(units=1))

        if model_spec["loss"] == "custom":
            loss = model_spec["custom_loss"]
        else:
            loss = model_spec["loss"]

        if model_spec["optimizer"] == "Adam":
            model.compile(
                optimizer=Adam(learning_rate=model_spec["learning_rate"]),
                loss=loss,
            )
        elif model_spec["optimizer"] == "Adagrad":
            model.compile(
                optimizer=Adagrad(learning_rate=model_spec["learning_rate"]),
                loss=loss,
            )
        elif model_spec["optimizer"] == "Nadam":
            model.compile(
                optimizer=Nadam(learning_rate=model_spec["learning_rate"]),
                loss=loss,
            )
        else:
            raise Exception("optimizer must be 'Adam', 'Adagrad' or 'Nadam'")

        return model

    def train_and_evaluate(
        self,
        train_X,
        train_Y,
        val_X,
        val_Y,
        conv_layers,
        filter_dim,
        pooling_dim,
        fc_layers,
        dropout_val,
        optimizer,
        learning_rate,
        batch_size,
        patience=5,
        max_epochs=100,
    ):
        model_spec = {
            "conv_layers": conv_layers,
            "filter_dim": filter_dim,
            "pooling_dim": pooling_dim,
            "fc_layers": fc_layers,
            "dropout_val": dropout_val,
            "input_shape": (train_X.shape[1], train_X.shape[2], 3),
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "loss": "binary_crossentropy",
        }
        model = Utils.build_model(model_spec)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )

        model.fit(
            train_X,
            train_Y,
            epochs=max_epochs,
            batch_size=batch_size,
            validation_data=(val_X, val_Y),
            callbacks=[early_stopping],
            verbose=0,
        )

        val_loss = model.evaluate(val_X, val_Y, verbose=0)[0]
        return val_loss

    def hyperparameter_tuning(
        self,
        optimizer_list,
        lr_list,
        batch_list,
        train_X,
        val_X,
        train_Y,
        val_Y,
    ):
        best_loss = float("inf")
        best_params = {}

        for optimizer in optimizer_list:
            for learning_rate in lr_list:
                for batch_size in batch_list:
                    for _ in range(3):
                        print(
                            f"Training with optimizer={optimizer}, learning_rate={learning_rate}, batch_size={batch_size}"
                        )
                        loss = Utils.train_and_evaluate(
                            train_X=train_X,
                            train_Y=train_Y,
                            val_X=val_X,
                            val_Y=val_Y,
                            conv_layers=[32, 64, 128],
                            filter_dim=(3, 3),
                            pooling_dim=(2, 2),
                            fc_layers=[512],
                            dropout_val=0.5,
                            optimizer=optimizer,
                            learning_rate=learning_rate,
                            batch_size=batch_size,
                        )
                        print(f"Loss: {loss}")
                        if loss < best_loss:
                            best_loss = loss
                            best_params = {
                                "optimizer": optimizer,
                                "learning_rate": learning_rate,
                                "batch_size": batch_size,
                            }
        print(f"Best loss: {best_loss}")
        print(f"Best Hyperparameters: {best_params}")
        return best_loss, best_params

    def architecture_tuning(
        self,
        best_params,
        conv_layers_list,
        filter_dim_list,
        pooling_dim_list,
        fc_layers_list,
        dropout_val_list,
        train_X,
        val_X,
        train_Y,
        val_Y,
    ):
        best_loss = float("inf")
        best_architecture = []
        for conv_layers_architecture in conv_layers_list:
            for filter_dim in filter_dim_list:
                for pooling_dim in pooling_dim_list:
                    for fc_layers_architecture in fc_layers_list:
                        for dropout_val in dropout_val_list:
                            for _ in range(3):
                                print(
                                    f"""Testing architecture:\n 
                                      Conv Layers: {conv_layers_architecture}\n 
                                      Filter Dim: {filter_dim}\n 
                                      Pooling Dim: {pooling_dim}\n 
                                      Fully Connected Layers: {fc_layers_architecture}\n 
                                      Dropout Val: {dropout_val}"""
                                )
                                loss = Utils.train_and_evaluate(
                                    train_X=train_X,
                                    train_Y=train_Y,
                                    val_X=val_X,
                                    val_Y=val_Y,
                                    conv_layers=conv_layers_architecture,
                                    filter_dim=filter_dim,
                                    pooling_dim=pooling_dim,
                                    fc_layers=fc_layers_architecture,
                                    dropout_val=dropout_val,
                                    optimizer=best_params["optimizer"],
                                    learning_rate=best_params["learning_rate"],
                                    batch_size=best_params["batch_size"],
                                )
                                print(f"loss: {loss}")
                                if loss < best_loss:
                                    best_loss = loss
                                    best_architecture = {
                                        "conv_layers": conv_layers_architecture,
                                        "filter_dim": filter_dim,
                                        "pooling_dim": pooling_dim,
                                        "fc_layers": fc_layers_architecture,
                                        "dropout_val": dropout_val,
                                    }

        print(f"Best loss: {best_loss}")
        print(f"Best architecture: {best_architecture}")

        return best_architecture
