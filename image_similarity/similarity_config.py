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
CHROMA_INSERT_BATCH = 5000

PACKAGE_NAME = "image_similarity"
ENCODER_MODEL_NAME = "deep_encoder.pt"    # 编码器权重保存路径（需写权限）
DECODER_MODEL_NAME = "deep_decoder.pt"    # 解码器权重保存路径（需写权限）
CHROMA_BACKEND_PATH = "chroma_backend"