__all__ = [
    'create_embeddings',
    'get_embedding_collection',
    'search_similar_img_ids'
]
from PIL import Image
import os
import re
import torchvision.transforms.transforms as T
import torch
import numpy

from image_similarity import similarity_model
from image_similarity import similarity_config
import chromadb
from chromadb import Documents,EmbeddingFunction,Embeddings
from math import ceil

def sorted_alphanumeric(data):
    convert = lambda  text:int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key:[convert(c) for  c in re.split('([0-9]+)', key)]
    return sorted(data,key=alphanum_key)
def get_id2image(main_dir,transform):
    all_images = sorted_alphanumeric(os.listdir(main_dir))
    id2imgs = {}
    for i,img in enumerate(all_images):
        img_loc = os.path.join(main_dir,img)
        image = Image.open(img_loc).convert("RGB")
        tensor_image = transform(image)
        id2imgs[str(i)] = tensor_image.numpy()
        return id2imgs

class MyEmbeddungFunction(EmbeddingFunction):
    def __init__(self,model):
        self.model = model
        model.to("cpu")
    def __call__(self,input:Documents) ->Embeddings:
        with torch.no_grad():
            return self.model(torch.tensor(numpy.array(input))).squeeze(0).numpy()


def create_embeddings(encoder):
     transform = T.Compose([
         T.Resize((64,64)),
         T.ToTensor()
     ])
     print("正在加载图片")
     id2imgs = get_id2image("../common/dataset", transform)
     ids = list(id2imgs.keys())
     imgs = list(id2imgs.values())
     print("图片加载完毕")
     print("正在向嵌入数据库中写入向量")
     chroma_client = chromadb.PersistentClient(similarity_config.CHROMA_BACKEND_PATH)
     collection = chroma_client.get_or_create_collection(
         name="image_similarity",
         embedding_function=MyEmbeddungFunction(encoder)
     )
     insert_batch_size = 5000
     for i in range(ceil(len(ids)/insert_batch_size)):
         collection.upsert(
             ids = ids[i*insert_batch_size:min(len(ids),(i+1)*insert_batch_size)],
             images=imgs[i*insert_batch_size:min(len(ids),(i+1)*insert_batch_size)]
         )
     print("向量写入完成")

def get_embedding_collection(encoder):
    chroma_client = chromadb.PersistentClient(similarity_config.CHROMA_BACKEND_PATH)
    return chroma_client.get_or_create_collection(
        name="image_similarity",
        embedding_function=MyEmbeddungFunction(encoder)
    )

def search_similar_img_ids(collection,image_tensor,img_cnt):
    result = collection.query(
        query_images = [image_tensor.numpy()],
        n_results = img_cnt,
    )
    ids = [int(id) for id in result["ids"][0]]
    return ids

if __name__ == "__main__":
    print("正在加载嵌入模型")
    encoder = similarity_model.ConvEncoder()
    encoder.load_state_dict(
        torch.load(
            os.path.join(
                '..',
                similarity_config.PACKAGE_NAME,
                similarity_config.ENCODER_MODEL_NAME
            )))
    print("嵌入模型加载完毕")
    collection = get_embedding_collection(encoder)
    print(collection.peek())
    transform = T.Compose([T.Resize((64, 64)),T.ToTensor()])
    test_img_path = os.path.join(similarity_config.IMG_PATH, "6582.jpg")
    test_img = Image.open(test_img_path).convert("RGB")
    test_img_tensor = transform(test_img)

    ids = search_similar_img_ids(collection, test_img_tensor, 5)
    print(f"{ids = }")













