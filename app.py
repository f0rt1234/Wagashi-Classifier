import keras
import numpy as np
from keras.models import load_model
import gradio as gr
from PIL import Image

classes = ["Castella", "Daifuku", "Dango", "Dorayaki", "Imagawayaki", "Taiyaki", "Youkan", "monaka"]
num_classes = len(classes)
examples = [["./food1.jpg"], ["./food1.jpg"]]
article = "<p style='text-align: center'>App created by Name F in 2024</p>"

def classify_food(input_img):
    # 画像の前処理
    image = input_img.convert('RGB')
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # モデルの読み込みと予測
    model = keras.models.load_model("./cnn_4.h5")
    prd = model.predict(image_array)

    # 予測結果の出力
    pred_class = classes[np.argmax(prd)]
    pred_prob = {classes[i]: prd[0][i] for i in range(len(classes))}
    return pred_class, pred_prob

# Gradioインターフェースの設定
gr.Interface(
    classify_food,
    inputs=gr.Image(label="Select input image", type="pil"),
    outputs=[gr.Label(label="Predicted Class"), gr.Label(label="Prediction Probabilities")],
    title="Wagashi Classifier",
    article=article,
    examples=examples,
).launch()
