import cv2
import os
import csv
import pandas as pd

# Directories and file paths
data_dir = "training_data"
csv_file = "labels.csv"
img_height, img_width = 400, 400

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.isfile(csv_file):
    df = pd.DataFrame(columns=["filename", "label"])
    df.to_csv(csv_file, index=False)


def capture_image():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Capture Image", frame)
        if cv2.waitKey(1) & 0xFF == ord("c"):
            img_name = f"image_{int(cv2.getTickCount())}.jpg"
            img_path = os.path.join(data_dir, img_name)
            cv2.imwrite(img_path, frame)
            break
    cap.release()
    cv2.destroyAllWindows()
    return img_name, img_path


def label_image():
    while True:
        label = (
            input(
                "Enter the label for the image (1 = biting/ 0 = not_biting): "
            )
            .strip()
            .lower()
        )
        if label in ["1", "0"]:
            break
        else:
            print(
                "Invalid input. Please enter '1 = biting' or '0 = not_biting'."
            )
    return 1 if label == "1" else 0


def append_to_csv(filename, label):
    df = pd.read_csv(csv_file)
    new_entry = pd.DataFrame(
        [[filename, label]], columns=["filename", "label"]
    )
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(csv_file, index=False)


def count_labels_csv(filename):
    count_0 = 0
    count_1 = 0

    with open(filename, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["label"] == "0":
                count_0 += 1
            elif row["label"] == "1":
                count_1 += 1
    print(f"Number of Not_Biting images in dataset: {count_0}")
    print(f"Number of Biting images in dataset: {count_1}")


def main():
    img_name, img_path = capture_image()
    label = label_image()
    append_to_csv(img_name, label)
    print(f"Image saved as {img_name} and labeled as '{label}'")
    count_labels_csv("labels.csv")


if __name__ == "__main__":
    main()
