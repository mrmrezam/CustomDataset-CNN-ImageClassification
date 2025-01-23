import os
import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from keras import models
from PIL import Image, ImageTk, ImageEnhance

model_path = "FinalModel64.keras"
cnnModel = models.load_model(model_path)

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = cnnModel.predict(img)
    return prediction

def show_prediction_plot():
    if current_prediction is not None:
        plt.bar(range(len(current_prediction[0])), current_prediction[0])
        plt.xlabel("Class Index")
        plt.ylabel("Probability")
        plt.title("Prediction Probability")
        plt.show()
    else:
        messagebox.showerror("Error", "No prediction available. Please upload an image first.")

def upload_image():
    global current_prediction
    file_path = filedialog.askopenfilename(title="Select Image File")
    if file_path:
        prediction = predict_image(file_path)
        current_prediction = prediction
        predicted_class_index = np.argmax(prediction)
        dataset_path = "dataset\\"
        class_names = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]

        predicted_class_name = class_names[predicted_class_index]

        result_label.config(text=f"The predicted class for the image is: {predicted_class_name}")
        result_label.place(x=100, y=250)

        image = Image.open(file_path)
        image = image.resize((300, 200))
        image = ImageTk.PhotoImage(image)
        img_label.config(image=image)
        img_label.image = image

root = Tk()
root.title("Image Classifier")
root.geometry("600x400")
root.configure(bg='lightblue')

bg_image = Image.open("sematec.jpg")
bg_image = bg_image.convert("L")
bg_image = bg_image.resize((600, 400))
enhancer = ImageEnhance.Brightness(bg_image)
bg_image = enhancer.enhance(0.2)
bg_image = ImageTk.PhotoImage(bg_image)
bg_label = Label(root, image=bg_image)
bg_label.place(relwidth=1, relheight=1)

upload_button = Button(root, text="Upload Image", command=upload_image, bg="lightgreen", fg="black", font=("Arial", 14, "bold"))
upload_button.place(x=100, y=55)

plot_button = Button(root, text="Show Prediction Plot", command=show_prediction_plot, bg="lightblue", fg="black", font=("Arial", 14, "bold"))
plot_button.place(x=350, y=300)

result_label = Label(root, text="Prediction will be shown here", font=("Arial", 14), bg='lightblue', fg="black")
result_label.place(x=100, y=250)

img_label = Label(root)
img_label.place(x=290, y=10)

root.mainloop()