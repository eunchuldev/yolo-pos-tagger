from functools import partial

from tensorflow.keras import Model, metrics, optimizers

from dataset import load_dataset
from model import loss, model


def train(
    dataset_pattern,
    name=None,
    channels=3,
    classes=2,
    anchors=[2, 4, 6],
    iou_threshold=0.5,
    focal_loss_gamma=0.0,
    learning_rate=1e-3,
    batch_size=64,
    epochs=2,
    validation_split=0.2,
):
    dataset, text_vocab, label_vocab = load_dataset(dataset_pattern, anchors=anchors)
    model_ = model(
        name, 
        channels=len(text_vocab),
        classes=len(label_vocab),
        anchors=anchors,
        )
    loss_ = partial(
        loss, iou_threshold=iou_threshold, focal_loss_gamma=focal_loss_gamma
    )
    model_.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate, clipnorm=0.001),
        loss=loss_,
        #metrics=[metrics.SparseCategoricalAccuracy()],
    )
    history = model_.fit(
        dataset,
        batch_size=batch_size,
        epochs=epochs,
    )
