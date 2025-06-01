# Disable oneDNN custom operations to suppress the log message
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import ctypes
ctypes.windll.user32.SetProcessDPIAware()  # Handle high-DPI displays

from keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import numpy as np

try:
    model = load_model('mnist.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def predict_digit(img):
    img = img.resize((28, 28))
    img = img.convert('L')
    img_array = np.array(img)
    img_array = 255 - img_array  # Invert colors (black-on-white -> white-on-black)
    Image.fromarray(img_array).save("debug.png")  # Save processed image
    img_array = img_array.reshape(1, 28, 28, 1)
    img_array = img_array / 255.0  # Normalize
    res = model.predict(img_array)[0]
    print("Prediction probabilities:", res)  # Debug
    return np.argmax(res), max(res)

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="DRAW..", font=("Arial", 48))
        self.classify_btn = tk.Button(self, text="Recognize", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)

        self.canvas.grid(row=0, column=0, pady=2, sticky=W)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete('all')
        self.label.configure(text="DRAW..")
        self.update()

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()
        rect = win32gui.GetWindowRect(HWND)
        a, b, c, d = rect
        rect = (a, b, c, d)  # Remove offset for full canvas capture
        try:
            im = ImageGrab.grab(rect)
        except Exception as e:
            print(f"ImageGrab error: {e}")
            self.label.configure(text="Capture failed")
            return
        im.save("captured.png")  # Save raw capture
        img_array = np.array(im.convert('L'))
        # Lower threshold to detect sparse drawings
        if img_array.sum() > 255 * 300 * 300 * 0.95:  # Adjusted for 300x300 canvas
            print(f"Canvas is empty (sum: {img_array.sum()})")
            self.label.configure(text="DRAW..")
            return
        digit, acc = predict_digit(im)
        print(f"Predicted: {digit}, Confidence: {int(acc*100)}%")
        self.label.configure(text=f"{digit},{int(acc*100)}%")
        self.update()

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 15  # Thicker brush
        self.canvas.create_oval(self.x-r, self.y-r, self.x+r, self.y+r, fill="black")

app = App()
mainloop()