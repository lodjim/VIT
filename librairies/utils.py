import pickle
import numpy as np

from loguru import logger
from rich.progress import track
import glob

import cv2
from PIL import Image
import torch.nn as nn
import torch as th
from torch.utils.data import TensorDataset, DataLoader

from .vit import ViT


def extract_features(path_to_data, path_to_save):
    data_acc = []
    labels_acc = []
    label = 0
    sub_directory = glob.glob(f'{path_to_data}/*')
    for directory in sub_directory:
        for path in track(glob.glob(f'{directory}/*'), directory):
            image = cv2.imread(path)
            img = cv2.resize(image, (384, 384), interpolation=cv2.INTER_CUBIC)
            data_acc.append(img)
            labels_acc.append(label)
        # end for
        label += 1
    # end for
    dataframe = {
        "data": data_acc,
        "labels": labels_acc,
    }
    with open(path_to_save, "wb") as fp:
        pickle.dump(dataframe, fp)
    logger.success(f'the data has been saved at {path_to_save}')


def load_data(path_to_dataset, batch_size):
    dataframe = pickle.load(open(path_to_dataset, 'rb'))
    data = dataframe['data']
    x_numpy = np.array(data,dtype=np.float32)
    x_tensor = th.from_numpy(x_numpy)
    x_tensor = th.reshape(x_tensor, (len(data), 3, 384, 384))
    y_numpy = np.array(dataframe['labels'])
    y_tensor = th.tensor(y_numpy, dtype=th.long)

    dataset = TensorDataset(x_tensor, y_tensor)
    dataset_for_model = DataLoader(
        dataset, batch_size=batch_size, shuffle=True)

    return dataset_for_model


def train_model(path_to_dataset, path_to_save, batch_size, epochs, lr):
    model = ViT()
    optimiser = th.optim.Adam(model.parameters(), lr=lr)
    dataset_for_model = load_data(path_to_dataset, batch_size)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        try:
            for batch_data, batch_labels in track(dataset_for_model, f"{epoch+1}/{epochs}"):
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
        except KeyboardInterrupt:
            logger.debug(f'training is interrupted,saved at {path_to_save}')
            break
        # end for
        logger.debug(f"Epoch{epoch+1}/{epochs},loss: {loss.item():.4f}")
    # end for
    logger.success('done.......')
    th.save(model.state_dict(), path_to_save)
    logger.success(f'the model is saved')

def predict(path_to_model,path_to_image):

    model = ViT()
    model.load_state_dict(th.load(
        path_to_model))
    model.eval()
    image = cv2.imread(path_to_image)
    img = cv2.resize(image, (384, 384), interpolation=cv2.INTER_CUBIC)
    img = (np.array(Image.fromarray(img))/128)-1
    img = th.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(th.float32)
    print(model(img))
