from tkinter import *
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("digit_model.h5")

root = Tk()
root.title("Handwritten Digit Predictor")

width = 280
height = 280

canvas = Canvas(root, width=width, height=height, bg="white")
canvas.pack()

image = Image.new("L", (width, height), "white")
draw = ImageDraw.Draw(image)

def paint(event):
    x1 = event.x - 8
    y1 = event.y - 8
    x2 = event.x + 8
    y2 = event.y + 8

    canvas.create_oval(
        x1, y1, x2, y2,
        fill="black",
        outline="black"
    )

    draw.ellipse(
        [x1, y1, x2, y2],
        fill="black"
    )

canvas.bind("<B1-Motion>", paint)

def predict():
    img = image.resize((28, 28))

    img_array = np.array(img)

    img_array = 255 - img_array

    img_array = img_array / 255.0

    img_array = img_array.reshape(1, 28, 28)

    prediction = model.predict(img_array)

    digit = np.argmax(prediction)

    result_label.config(
        text=f"Predicted Digit: {digit}"
    )

def clear_canvas():
    canvas.delete("all")

    draw.rectangle(
        [0, 0, width, height],
        fill="white"
    )

    result_label.config(
        text="Draw a Digit"
    )

Button(root, text="Predict", command=predict).pack()
Button(root, text="Clear", command=clear_canvas).pack()

result_label = Label(
    root,
    text="Draw a Digit",
    font=("Arial", 16)
)

result_label.pack()

root.mainloop()