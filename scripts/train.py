"""Script used for model training."""

import os.path
import matplotlib.pyplot as plt
import torch
import configs.train_config as config

from tqdm import tqdm
from scripts.utils import get_dataloader, save_plot
from scripts.multiclass_model import MultiClassClassifier


def train_loop(dataloader, model, loss_fn, optimizer):
    """
    Perform training loop on all training dataset and return average loss and average accuracy.
    """
    dataset_size, n_batches = len(dataloader.dataset), len(dataloader)
    total_loss, total_acc = 0, 0

    model.train()
    for batch, (X, y) in tqdm(enumerate(dataloader)):
        predictions = model(X)
        loss = loss_fn(predictions, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss
        total_acc += (predictions.argmax(1) == y).type(torch.float).sum().item()

    avg_loss = total_loss / n_batches
    avg_acc = total_acc / dataset_size
    print(f"Train Error: \n Accuracy: {(100 * avg_acc):>0.1f}%, Avg loss: {avg_loss:>8f} \n")

    return avg_loss, avg_acc


def test_loop(dataloader, model, loss_fn):
    """
    Perform test loop on validation/test dataset and return average loss and average accuracy.
    """

    dataset_size, n_batches = len(dataloader.dataset), len(dataloader)
    total_loss, total_acc = 0, 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            predictions = model(X)
            loss = loss_fn(predictions, y)
            total_loss += loss
            total_acc += (predictions.argmax(1) == y).type(torch.float).sum().item()

    avg_loss = total_loss / n_batches
    avg_acc = total_acc / dataset_size

    print(f"Test Error: \n Accuracy: {(100 * avg_acc):>0.1f}%, Avg loss: {avg_loss:>8f} \n")
    return avg_loss, avg_acc


def perform_learning(epochs,
                     train_loader,
                     val_loader,
                     save_folder,
                     model,
                     loss_fn,
                     optimizer,
                     save_current_best: bool):

    history = {'train_loss': [],
               'train_acc': [],
               'val_loss': [],
               'val_acc': []}

    for epoch in tqdm(range(epochs)):

        train_loss, train_acc = train_loop(train_loader, model, loss_fn, optimizer)
        val_loss, val_acc = test_loop(val_loader, model, loss_fn)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        if save_current_best and val_loss <= min(history['val_loss']):
            torch.save(model, os.path.join(save_folder, 'model.pth'))

    if not save_current_best:
        torch.save(model, os.path.join(save_folder, 'model.pth'))

    for key in ['acc', 'loss']:
        save_plot(history=history,
                  key=key,
                  output_path=os.path.join(config.SAVE_MODEL_PATH, f'{key}.png'))

    return history


if __name__ == '__main__':

    train_dataset, train_loader = get_dataloader(root_dir=config.TRAIN_PATH,
                                                 transforms=config.TRAIN_TRANSFORM,
                                                 target_transform=None,
                                                 batch_size=config.BATCH_SIZE)
    _, val_loader = get_dataloader(root_dir=config.VAL_PATH,
                                   transforms=config.VALIDATION_TRANSFORM,
                                   target_transform=None,
                                   batch_size=config.BATCH_SIZE,
                                   shuffle=False)

    net = MultiClassClassifier(class_number=config.NUM_CLASSES)
    loss_function = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=config.LR)

    perform_learning(epochs=1,
                     train_loader=train_loader,
                     val_loader=val_loader,
                     save_folder=config.SAVE_MODEL_PATH,
                     model=net,
                     loss_fn=loss_function,
                     optimizer=opt,
                     save_current_best=True
                     )







