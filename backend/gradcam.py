import tensorflow as tf
import numpy as np
import cv2

def generate_gradcam(model, img_array, class_index):
    base_model = model.get_layer("efficientnetb3")

    with tf.GradientTape() as tape:
        conv_outputs = base_model(img_array)
        tape.watch(conv_outputs)

        x = model.layers[-8](conv_outputs)
        x = model.layers[-7](x)
        x = model.layers[-6](x)
        x = model.layers[-5](x)
        x = model.layers[-4](x)
        x = model.layers[-3](x)
        preds = model.layers[-1](x)

        loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    heatmap = cv2.resize(heatmap.numpy(), (300, 300))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img = img_array[0]
    img = ((img + 1) * 127.5).astype(np.uint8)

    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
