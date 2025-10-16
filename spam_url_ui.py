import tkinter as tk
from tkinter import messagebox
import joblib

# Load models
vectorizer = joblib.load("vectorizer.pkl")
svm_model = joblib.load("svm_model.pkl")
rf_model = joblib.load("rf_model.pkl")

def predict_url():
    url = url_entry.get()
    if not url.strip():
        messagebox.showwarning("Input Error", "Please enter a URL.")
        return
    
    features = vectorizer.transform([url])
    pred_svm = svm_model.predict(features)[0]
    pred_rf = rf_model.predict(features)[0]

    result = f"SVM: {'SPAM' if pred_svm else 'SAFE'}\nRandom Forest: {'SPAM' if pred_rf else 'SAFE'}"
    messagebox.showinfo("Prediction Result", result)

# UI Setup
root = tk.Tk()
root.title("Spam URL Detection")
root.geometry("400x250")

tk.Label(root, text="Enter URL to Check:", font=("Arial", 12)).pack(pady=10)
url_entry = tk.Entry(root, width=50)
url_entry.pack(pady=5)

tk.Button(root, text="Predict", command=predict_url, bg="#007BFF", fg="white", font=("Arial", 12)).pack(pady=15)
tk.Button(root, text="Exit", command=root.quit, bg="red", fg="white").pack()

root.mainloop()
