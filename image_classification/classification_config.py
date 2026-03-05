
FASHION_LABELS_PATH = "../common/fashion-labels.csv"
IMG_PATH = "../common/dataset"
IMG_WEIGHT = 64
IMG_WIDTH = 64


SEED = 42
TRAIN_RATIO = 0.75
VAL_RATIO = 1-TRAIN_RATIO
SHUFFLE_BUFFER_SIZE = 100

LEARNING_RATE = 1e-3
EPOCHS = 30
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
FULL_BATCH_SIZE = 32

PACKAGE_NAME = "image_classification"
CLASSIFIER_MODEL_NAME = "classifier.pt"

classification_names = {
    0: '上身衣服',  # 数字 0 对应“上身衣服”
    1: '鞋',       # 数字 1 对应“鞋”
    2: '包',       # 数字 2 对应“包”
    3: '下身衣服',  # 数字 3 对应“下身衣服”
    4: '手表'      # 数字 4 对应“手表”
}

