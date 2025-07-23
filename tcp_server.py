import socket
import threading
import json
import os
import struct
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from torchvision import transforms
from torch.nn import functional as F
from models.registry import get_model
from threading import Semaphore

# ========== 读取配置 ==========
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

DEVICE = config["device"]
INPUT_SIZE = config["input_size"]
MAX_THREADS = config.get("max_threads", 4)
HOST = config.get("host", "0.0.0.0")
PORT = config.get("port", 5678)

# ========== 加载模型 ==========
print("🚀 加载模型中...")
model = get_model(config["model_name"], config["in_channels"], config["out_channels"]).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(config["save_dir"], config["save_filename"]), map_location=DEVICE))
model.eval()
print("✅ 模型加载完成")

# ========== 线程控制 ==========
thread_sema = Semaphore(MAX_THREADS)

# ========== 预测函数 ==========
def predict_from_bytes(img_bytes):
    image = Image.open(BytesIO(img_bytes)).convert("RGB")
    image = image.resize(INPUT_SIZE)
    tensor = transforms.ToTensor()(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(tensor)
        pred = F.interpolate(pred, size=INPUT_SIZE, mode="bilinear", align_corners=False)
        pred = torch.sigmoid(pred) if config["out_channels"] == 1 else torch.softmax(pred, dim=1)
        pred_np = pred.squeeze().cpu().numpy()
        if config["out_channels"] == 1:
            return (pred_np > 0.5).astype(np.uint8) * 255
        else:
            return np.argmax(pred_np, axis=0).astype(np.uint8)

# ========== 客户端处理 ==========
def handle_client(conn, addr):
    with thread_sema:
        try:
            print(f"[+] 接入客户端 {addr}")

            # 1️⃣ 发送模型信息
            model_info = {
                "input_size": config["input_size"],
                "channels": config["in_channels"],
                "classes": config["out_channels"]
            }
            info_bytes = json.dumps(model_info).encode("utf-8")
            conn.sendall(struct.pack("I", len(info_bytes)) + info_bytes)

            # 2️⃣ 接收图像数据长度
            data_len_bytes = conn.recv(4)
            if not data_len_bytes:
                print("[-] 未接收到数据长度，断开连接")
                return
            img_len = struct.unpack("I", data_len_bytes)[0]

            # 3️⃣ 接收图像数据
            img_data = b""
            while len(img_data) < img_len:
                packet = conn.recv(img_len - len(img_data))
                if not packet:
                    break
                img_data += packet

            if len(img_data) != img_len:
                print("[-] 接收不完整图像数据")
                return

            # 4️⃣ 预测
            mask = predict_from_bytes(img_data)

            # 5️⃣ 返回mask
            mask_img = Image.fromarray(mask)
            with BytesIO() as output_buf:
                mask_img.save(output_buf, format="PNG")
                result_bytes = output_buf.getvalue()

            conn.sendall(struct.pack("I", len(result_bytes)) + result_bytes)
            print(f"[✓] {addr} 预测完成并返回")

        except Exception as e:
            print(f"[!] 处理客户端 {addr} 出错：{e}")
        finally:
            conn.close()
            print(f"[-] 客户端 {addr} 断开连接")

# ========== 主监听函数 ==========
def start_server():
    print(f"🖥️ 服务器监听启动: {HOST}:{PORT} (最大线程数: {MAX_THREADS})")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        while True:
            conn, addr = s.accept()
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    start_server()
