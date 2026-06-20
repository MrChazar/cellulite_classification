import os
import cv2
import torch
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch.nn as nn
from torchvision import transforms, models


class ThermalHistogramAndGray(object):
    def __init__(self, clip_limit=3.0, grid_size=(15, 15)):
        self.clip_limit = clip_limit
        self.grid_size = grid_size

    def __call__(self, img):
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.grid_size)
        enhanced_gray = clahe.apply(gray)
        enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(enhanced_rgb)


def get_model(device, num_classes=4):
    model = models.convnext_tiny(weights=None)

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    return model.to(device)


class CelluliteApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Termograficzna Detekcja Cellulitu")
        self.root.geometry("550x700")

        self.bg_color = "#12002b"
        self.panel_bg = "#290054"
        self.text_main = "#ffffff"
        self.text_accent = "#ffcc00"
        self.btn_color = "#ff6600"
        self.btn_hover = "#ff8533"
        self.alert_color = "#e60073"
        self.success_color = "#00ffcc"

        self.root.configure(bg=self.bg_color)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_multi = None
        self.model_binary = None
        self.model_multi_path = "weights/multiclass.pth"
        self.model_binary_path = "weights/binary.pth"

        self.multi_class_names = {
            0: "Brak cellulitu (Stopień 0)",
            1: "Cellulit łagodny (Stopień 1)",
            2: "Cellulit umiarkowany (Stopień 2)",
            3: "Cellulit zaawansowany (Stopień 3)"
        }

        self.binary_class_names = {
            0: "Zdrowe (0)",
            1: "Cellulit (1)"
        }

        self.setup_ui()
        self.load_models()
        self.setup_transforms()

    def setup_ui(self):
        # Tytuł aplikacji
        title_lbl = tk.Label(
            self.root,
            text="WYKRYWANIE CELLULITU",
            font=("Segoe UI", 18, "bold"),
            bg=self.bg_color,
            fg=self.text_accent
        )
        title_lbl.pack(pady=(25, 10))

        self.btn_load = tk.Button(
            self.root,
            text="Wczytaj Skraj Termograficzny",
            font=("Segoe UI", 12, "bold"),
            bg=self.btn_color,
            fg=self.text_main,
            activebackground=self.btn_hover,
            activeforeground=self.text_main,
            relief="flat",
            cursor="hand2",
            padx=20,
            pady=10,
            command=self.load_image
        )
        self.btn_load.pack(pady=15)

        self.btn_load.bind("<Enter>", lambda e: self.btn_load.config(bg=self.btn_hover))
        self.btn_load.bind("<Leave>", lambda e: self.btn_load.config(bg=self.btn_color))

        self.image_frame = tk.Frame(
            self.root,
            bg=self.panel_bg,
            bd=2,
            relief="ridge",
            highlightbackground=self.btn_color,
            highlightthickness=2
        )
        self.image_frame.pack(pady=10, padx=40, fill="both", expand=True)

        self.panel = tk.Label(self.image_frame, bg=self.panel_bg, text="Brak obrazu", fg="#7a4b9c",
                              font=("Segoe UI", 12))
        self.panel.pack(expand=True)

        self.result_lbl = tk.Label(
            self.root,
            text="Oczekuje na wgranie danych...",
            font=("Segoe UI", 13),
            bg=self.bg_color,
            fg=self.text_main,
            justify="center"
        )
        self.result_lbl.pack(pady=(15, 25))

    def load_models(self):
        try:
            self.model_multi = get_model(self.device, num_classes=4)
            self.model_multi.load_state_dict(torch.load(self.model_multi_path, map_location=self.device))
            self.model_multi.eval()

            self.model_binary = get_model(self.device, num_classes=2)
            self.model_binary.load_state_dict(torch.load(self.model_binary_path, map_location=self.device))
            self.model_binary.eval()

            print("Pomyślnie załadowano oba modele.")
        except FileNotFoundError as e:
            messagebox.showerror("Błąd Systemu",
                                 f"Nie znaleziono pliku z wagami modelu: {e.filename}\nUpewnij się, że pliki znajdują się w katalogu roboczym.")
            self.btn_load.config(state=tk.DISABLED, bg="#555555")

    def setup_transforms(self):
        mean_imagenet = [0.485, 0.456, 0.406]
        std_imagenet = [0.229, 0.224, 0.225]

        self.transform_multi = transforms.Compose([
            transforms.Resize((240, 210)),
            ThermalHistogramAndGray(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_imagenet, std=std_imagenet)
        ])

        self.transform_binary = transforms.Compose([
            transforms.Resize((224, 224)),
            ThermalHistogramAndGray(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_imagenet, std=std_imagenet)
        ])

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Wybierz termogram",
            filetypes=[("Pliki obrazów", "*.jpg *.jpeg *.png")]
        )

        if not file_path:
            return

        img_display = Image.open(file_path)
        img_display.thumbnail((380, 380))  # Dopasowanie do nowej ramki
        img_tk = ImageTk.PhotoImage(img_display)

        self.panel.configure(image=img_tk, text="")
        self.panel.image = img_tk

        self.predict(file_path)

    def predict(self, image_path):
        self.result_lbl.config(text="Analizowanie tkanki...", fg=self.text_accent)
        self.root.update()

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)

        input_multi = self.transform_multi(image_pil).unsqueeze(0).to(self.device)
        input_binary = self.transform_binary(image_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs_multi = self.model_multi(input_multi)
            prob_multi = torch.nn.functional.softmax(outputs_multi, dim=1)
            conf_multi, pred_multi = torch.max(prob_multi, 1)
            class_idx_multi = pred_multi.item()
            conf_score_multi = conf_multi.item() * 100

            outputs_binary = self.model_binary(input_binary)
            prob_binary = torch.nn.functional.softmax(outputs_binary, dim=1)
            conf_binary, pred_binary = torch.max(prob_binary, 1)
            class_idx_binary = pred_binary.item()
            conf_score_binary = conf_binary.item() * 100

        result_text = (
            f"Klasyfikacja binarna: {self.binary_class_names[class_idx_binary]} ({conf_score_binary:.1f}%)\n"
            f"Klasyfikacja wieloklasowa: {self.multi_class_names[class_idx_multi]} ({conf_score_multi:.1f}%)"
        )

        color = self.success_color if class_idx_binary == 0 else self.alert_color
        self.result_lbl.config(text=result_text, fg=color)


if __name__ == "__main__":
    root = tk.Tk()
    app = CelluliteApp(root)
    root.mainloop()