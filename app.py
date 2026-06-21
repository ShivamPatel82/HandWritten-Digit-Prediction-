import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# Load model
model = tf.keras.models.load_model("digit_model.h5")

st.title("✍️ Handwritten Digit Predictor")

st.write("Canvas पर 0-9 तक कोई digit लिखें और Predict दबाएँ")

# Canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):

    if canvas_result.image_data is not None:

        img = Image.fromarray(
            canvas_result.image_data.astype("uint8")
        )

        img = img.convert("L")

        img = img.resize((28, 28))

        img_array = np.array(img)

        img_array = 255 - img_array

        img_array = img_array / 255.0

        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = model.predict(img_array)

        digit = np.argmax(prediction)

        confidence = np.max(prediction) * 100

        st.success(
            f"Predicted Digit: {digit}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )