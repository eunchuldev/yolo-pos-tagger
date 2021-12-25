import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from tensor_annotations import axes
from tensorflow.keras import Model
from tensorflow.keras.layers import (Add, BatchNormalization, Conv1D, Input,
                                     LeakyReLU, MaxPool1D, Reshape,
                                     UpSampling1D, ZeroPadding1D)
from tensorflow.keras.losses import (binary_crossentropy,
                                     sparse_categorical_crossentropy)
from tensorflow.keras.regularizers import l2

Batch = axes.Batch
Width = axes.Width
Channels = axes.Channels


class Anchors(axes.Axis):
    pass


class XW(axes.Axis):
    pass

def conv(
    x: ttf.Tensor3[Batch, Width, Channels],
    filters,
    kernel_size=3,
    strides=1,
    batch_norm=True,
    activate=True,
) -> ttf.Tensor3[Batch, Width, Channels]:
    if strides == 1:
        padding = "same"
    else:
        padding = "valid"
        x = ZeroPadding1D(padding=1)(x)  # top left half-padding
    x = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=not batch_norm,
        kernel_regularizer=l2(0.0005),
    )(x)
    if batch_norm:
        x = BatchNormalization()(x)
    if activate:
        x = LeakyReLU(alpha=0.1)(x)
    return x


def residual(
    x: ttf.Tensor3[Batch, Width, Channels], filters
) -> ttf.Tensor3[Batch, Width, Channels]:
    shortcut = x
    x = conv(x, filters, 1)
    x = conv(x, filters, 3)
    x = Add()([shortcut, x])
    return x


def backbone(
    x: ttf.Tensor3[Batch, Width, Channels]
) -> ttf.Tensor3[Batch, Width, Channels]:
    x = conv(x, 256, kernel_size=1)
    x = residual(x, 256)
    x = residual(x, 256)
    x = residual(x, 256)
    x = residual(x, 256)
    x = residual(x, 256)
    return x


def head(
    x: ttf.Tensor3[Batch, Width, Channels], classes: int, anchors: list[int]
) -> ttf.Tensor4[Batch, Width, Anchors, Channels]:
    x = conv(x, 128, kernel_size=1)
    x = conv(x, 256, kernel_size=3)
    x = conv(x, 128, kernel_size=1)
    x = conv(x, 256, kernel_size=3)
    x = conv(x, 128, kernel_size=1)
    x = conv(x, 256, kernel_size=3)
    # anchors * (x, w, objectivity, ...classes)
    x = conv(
        x,
        len(anchors) * (classes + 3),
        kernel_size=1,
        batch_norm=False,
        activate=False,
    )
    y = tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], len(anchors), classes + 3))
    tiles = tf.range(start=0, limit=tf.shape(y)[1], delta=1.0)
    tiles = tiles[..., tf.newaxis, tf.newaxis]
    anchors = tf.constant(anchors, tf.float32)[..., tf.newaxis]
    cell_x, cell_w, objectness, class_probs = tf.split(y, (1, 1, 1, -1), axis=-1)
    pixel_x = tf.sigmoid(cell_x)
    pixel_w = tf.exp(cell_w) * anchors + tiles
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    y = tf.concat([pixel_x, pixel_w, objectness, class_probs], axis=-1)
    return y


def model(name=None, channels=3, classes=2, anchors=[2, 4, 6], traning=False) -> Model:
    """
    Yolo inspired model

           g   o   o   d      g   u   y

    x          1   0              0.5
    w          2   2              1.5
    objec  1   1   1   1   0  1   1   1
    class      Adj Adj        N   N   N

    w = exp(w') * anchor
    output: anchors * (x, w`, objectivity, ...classes)
    """
    inputs: ttf.Tensor3[Batch, Width, Channels] = Input([None, channels])
    x = backbone(inputs)
    # output channels: (pixel_x, pixel_w, objectivity, ...classes)
    y: ttf.Tensor4[Batch, Width, Anchors, Channels] = head(x, classes, anchors)
    if traning:
        return tf.keras.Model(inputs, y, name=name)
    else:
        return tf.keras.Model(inputs, y, name=name)

def bbox_iou(
    xw_boxes1: ttf.Tensor3[Batch, Width, XW], xw_boxes2: ttf.Tensor3[Batch, Width, XW]
):
    # bbox: [Batch, Width, X1X2]
    lr_boxes1 = tf.concat(
        [
            xw_boxes1[..., :1] - xw_boxes1[..., 1:] * 0.5,
            xw_boxes1[..., :1] + xw_boxes1[..., 1:] * 0.5,
        ],
        axis=-1,
    )
    lr_boxes2 = tf.concat(
        [
            xw_boxes2[..., :1] - xw_boxes2[..., 1:] * 0.5,
            xw_boxes2[..., :1] + xw_boxes2[..., 1:] * 0.5,
        ],
        axis=-1,
    )
    left = tf.maximum(lr_boxes1[..., :1], lr_boxes2[..., :1])
    right = tf.minimum(lr_boxes1[..., 1:], lr_boxes2[..., 1:])
    inter_area = tf.maximum(right - left, 0.0)
    union_area = xw_boxes1[..., 1:2] + xw_boxes2[..., 1:2] - inter_area
    return 1.0 * inter_area / union_area


def bbox_giou(
    xw_boxes1: ttf.Tensor3[Batch, Width, XW], xw_boxes2: ttf.Tensor3[Batch, Width, XW]
):
    lr_boxes1 = tf.concat(
        [
            xw_boxes1[..., :1] - xw_boxes1[..., 1:] * 0.5,
            xw_boxes1[..., :1] + xw_boxes1[..., 1:] * 0.5,
        ],
        axis=-1,
    )
    lr_boxes2 = tf.concat(
        [
            xw_boxes2[..., :1] - xw_boxes2[..., 1:] * 0.5,
            xw_boxes2[..., :1] + xw_boxes2[..., 1:] * 0.5,
        ],
        axis=-1,
    )
    left = tf.maximum(lr_boxes1[..., :1], lr_boxes2[..., :1])
    right = tf.minimum(lr_boxes1[..., 1:], lr_boxes2[..., 1:])
    inter_area = tf.maximum(right - left, 0.0)
    union_area = xw_boxes1[..., 1:2] + xw_boxes2[..., 1:2] - inter_area
    iou = 1.0 * inter_area / union_area

    enclose_left = tf.minimum(lr_boxes1[..., :1], lr_boxes2[..., :1])
    enclose_right = tf.maximum(lr_boxes1[..., 1:], lr_boxes2[..., 1:])
    enclose_area = tf.maximum(enclose_right - enclose_left, 0.0)
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou


def loss(
    pred: ttf.Tensor4[Batch, Width, Anchors, Channels],
    label: ttf.Tensor4[Batch, Width, Anchors, Channels],
    iou_threshold=0.5,
    focal_loss_gamma=0.0,
) -> ttf.Tensor1[Batch]:
    pred_xw_boxes, pred_objectness, pred_classes = tf.split(pred, (2, 1, -1), axis=-1)
    label_xw_boxes, label_objectness, label_classes = tf.split(
        label, (2, 1, -1), axis=-1
    )
    giou = bbox_giou(pred_xw_boxes, label_xw_boxes)
    #bbox_loss_scale = 2.0 - 1.0 * (label_xw_boxes[..., 1:2] / float(tf.shape(pred)[1]))
    bbox_loss_scale = 1.0
    giou_loss = label_objectness * bbox_loss_scale * (1 - giou)

    iou = bbox_iou(pred_xw_boxes, label_xw_boxes)
    max_iou = tf.reduce_max(iou, axis=-1)

    focal_scale = tf.squeeze(tf.pow(label_objectness - pred_objectness, focal_loss_gamma), -1)
    obj_mask = tf.squeeze(label_objectness, -1)
    ignore_mask = tf.cast(max_iou < iou_threshold, tf.float32)
    obj_loss = binary_crossentropy(label_objectness, pred_objectness)
    objectness_loss = focal_scale * (
        obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss
    )

    class_loss = obj_mask * binary_crossentropy(pred_classes, label_classes)

    giou_loss = tf.reduce_sum(giou_loss, axis=[1, 2, 3])
    objectness_loss = tf.reduce_sum(objectness_loss, axis=[1, 2])
    class_loss = tf.reduce_sum(class_loss, axis=[1, 2])

    return giou_loss + class_loss + objectness_loss

