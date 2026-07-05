import os
import random
import shutil

# ----------------------------
# Dataset Paths
# ----------------------------

SOURCE_DIR = "dataset/original"

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/validate"
TEST_DIR = "dataset/test"

# Split ratio

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

# ----------------------------
# Create folders
# ----------------------------

for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:

    os.makedirs(folder, exist_ok=True)

    for cls in ["Drowsy", "Non Drowsy"]:

        os.makedirs(os.path.join(folder, cls), exist_ok=True)


# ----------------------------
# Split Images
# ----------------------------

for cls in ["Drowsy", "Non Drowsy"]:

    source_class = os.path.join(SOURCE_DIR, cls)

    images = os.listdir(source_class)

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)

    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    print(f"\n{cls}")
    print("Total :", total)
    print("Train :", len(train_images))
    print("Validation :", len(val_images))
    print("Test :", len(test_images))

    # Train

    for img in train_images:

        shutil.copy(
            os.path.join(source_class, img),
            os.path.join(TRAIN_DIR, cls, img)
        )

    # Validation

    for img in val_images:

        shutil.copy(
            os.path.join(source_class, img),
            os.path.join(VAL_DIR, cls, img)
        )

    # Test

    for img in test_images:

        shutil.copy(
            os.path.join(source_class, img),
            os.path.join(TEST_DIR, cls, img)
        )

print("\nDataset Split Completed Successfully!")