import numpy as np
from PIL import Image
from tensorflow.lite.python.interpreter import Interpreter

class Model:
    def __init__(self, model_path):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]['shape']  # e.g., [1, 3, 320, 320]
        self.floating_model = self.input_details[0]['dtype'] == np.float32

        self.input_mean = 0.0
        self.input_std = 255.0

    def prepare(self):
        pass

    def predict(self, image):
        # Resize ảnh về đúng kích thước
        height = self.input_shape[2]
        width = self.input_shape[3]
        image = image.resize((width, height)).convert("RGB")

        # Convert PIL to numpy
        input_data = np.array(image, dtype=np.float32 if self.floating_model else np.uint8)

        # Chuyển từ (H, W, C) → (C, H, W)
        input_data = np.transpose(input_data, (2, 0, 1))

        # Thêm batch dimension → [1, C, H, W]
        input_data = np.expand_dims(input_data, axis=0)

        # Normalize nếu cần
        if self.floating_model:
            input_data = (input_data - self.input_mean) / self.input_std

        # Gán dữ liệu vào model
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Lấy output: [1, num_classes, H, W]
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        output_data = np.squeeze(output_data)  # [num_classes, H, W]

        # Argmax để lấy segmentation map: [H, W]
        mask = np.argmax(output_data, axis=0).astype(np.uint8)
        return mask
