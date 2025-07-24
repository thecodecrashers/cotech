import sys
import os
import json
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QMessageBox,
    QStackedLayout,QTextEdit
)
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QScrollArea
from PyQt5.QtCore import QProcess
import platform
def launch_python_script(script_name: str):
    """
    启动脚本：用当前 Python 解释器来执行 script_name 文件
    会自动判断平台（Windows / Linux）
    """
    python_path = sys.executable
    script_path = os.path.abspath(script_name)

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"❌ 脚本未找到: {script_path}")

    try:
        if platform.system() == "Windows":
            subprocess.Popen(["start", "cmd", "/k", python_path, script_path], shell=True)
        else:
            subprocess.Popen(["x-terminal-emulator", "-e", python_path, script_path])
    except Exception as e:
        raise RuntimeError(f"❌ 脚本启动失败: {e}")

# ========================== 共用配置小部件 ==========================

class ConfigFragment(QWidget):
    def __init__(self, config_path, fields: list, label_map=None):
        super().__init__()
        self.config_path = config_path
        self.fields = fields
        self.label_map = label_map or {}
        self.inputs = {}
        self.layout = QVBoxLayout(self)
        self.load_config()
        self.build_ui()
        self.add_buttons()

    def load_config(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

    def get_value_by_path(self, path):
        keys = path.split('.')
        value = self.config
        for k in keys:
            value = value[k]
        return value

    def set_value_by_path(self, path, new_value):
        keys = path.split('.')
        obj = self.config
        for k in keys[:-1]:
            obj = obj[k]
        obj[keys[-1]] = new_value

    def build_ui(self):
        for key in self.fields:
            hbox = QHBoxLayout()
            label = QLabel(self.label_map.get(key, key))
            label.setFixedWidth(200)

            value = self.get_value_by_path(key)
            edit = QLineEdit(str(value))
            edit.setMinimumWidth(300)

            if "dir" in key or "path" in key:
                btn = QPushButton("📂")
                btn.setFixedWidth(40)

                def make_open_dialog(edit_widget, field_key):
                    def open_dialog():
                        try:
                            if "path" in field_key:
                                path, _ = QFileDialog.getSaveFileName(self, "选择文件", filter="所有文件 (*)")
                            else:
                                path = QFileDialog.getExistingDirectory(self, "选择文件夹")
                            if path:
                                edit_widget.setText(path)
                        except Exception as e:
                            QMessageBox.critical(self, "打开失败", f"打开文件/文件夹失败：\n{str(e)}")
                    return open_dialog

                btn.clicked.connect(make_open_dialog(edit, key))

                hbox.addWidget(label)
                hbox.addWidget(edit)
                hbox.addWidget(btn)
            else:
                hbox.addWidget(label)
                hbox.addWidget(edit)

            self.layout.addLayout(hbox)
            self.inputs[key] = edit

    def add_buttons(self):
        hbox = QHBoxLayout()
        save_btn = QPushButton("💾 保存配置")
        save_btn.clicked.connect(self.save_config)
        hbox.addWidget(save_btn)
        self.layout.addLayout(hbox)

    def save_config(self):
        for key, edit in self.inputs.items():
            raw = edit.text()
            try:
                val = eval(raw, {}, {})
            except:
                val = raw
            self.set_value_by_path(key, val)

        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

        QMessageBox.information(self, "保存成功", "配置已保存到 config.json")


# ========================== 主界面 ==========================

class MainUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("工业视觉训练平台")
        self.resize(1280,720)

        main_layout = QVBoxLayout(self)

        # 顶部按钮
        button_layout = QHBoxLayout()
        self.btn_annotate = QPushButton("🖼 图像标注")
        self.btn_data = QPushButton("📂 数据处理")
        self.btn_train = QPushButton("🧠 模型训练")
        self.btn_finetune = QPushButton("🔧 微调模型")
        self.btn_predict = QPushButton("📤 模型推理")
        self.btn_tcp_server = QPushButton("🌐 启动 TCP 服务器")

        for btn in [self.btn_annotate, 
                    self.btn_data, 
                    self.btn_train, 
                    self.btn_predict,
                    self.btn_finetune,
                    self.btn_tcp_server]:
            btn.setFixedHeight(40)
            button_layout.addWidget(btn)
        main_layout.addLayout(button_layout)

        # 页面堆叠
        self.stack = QStackedLayout()
        main_layout.addLayout(self.stack)

        # 创建并包裹页面
        self.page_data = self.wrap_scrollable_page(self.create_data_page())
        self.page_annotate = self.wrap_scrollable_page(self.create_annotate_page())
        self.page_train = self.wrap_scrollable_page(self.create_train_page())
        self.page_finetune = self.wrap_scrollable_page(self.create_finetune_page())
        self.page_predict = self.wrap_scrollable_page(self.create_predict_page())
        self.page_tcp_server = self.wrap_scrollable_page(self.create_tcp_server_page())

        self.stack.addWidget(self.page_annotate)
        self.stack.addWidget(self.page_data)
        self.stack.addWidget(self.page_train)
        self.stack.addWidget(self.page_predict)
        self.stack.addWidget(self.page_finetune)
        self.stack.addWidget(self.page_tcp_server)
        # 按钮绑定
        self.btn_data.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_data))
        self.btn_annotate.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_annotate))
        self.btn_train.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_train))
        self.btn_finetune.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_finetune))
        self.btn_predict.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_predict))
        self.btn_tcp_server.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_tcp_server))

    def wrap_scrollable_page(self, widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        return scroll


    def create_data_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        label = QLabel("👉 数据预处理功能（TODO: 分割、统计、增强、格式转换等）")
        label.setAlignment(Qt.AlignCenter)

        config = ConfigFragment(
            "config.json",
            fields=["annotate_img_dir",
                    "preprocess_train_ratio", 
                    "preprocess_val_ratio", 
                    "preprocess_test_ratio",
                    "preprocess_augment_times",
                    "preprocess_include_original"],
            label_map={"annotate_img_dir": "标注图像路径",
                       "preprocess_train_ratio": "训练集比例",
                       "preprocess_val_ratio": "验证集比例",
                       "preprocess_test_ratio": "测试集比例",
                       "preprocess_augment_times":"数据增强次数" ,
                       "preprocess_include_original":"包含原图"}
        )
        run_btn = QPushButton("▶ 运行数据处理脚本")
        def run_preprocessing():
            try:
                python_path = sys.executable  # 当前 PyQt 运行用的 Python
                script_path = os.path.abspath("utils/split_labeled_dataset.py")

                result = subprocess.run([python_path, script_path], check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True)

                QMessageBox.information(None, "完成", f"处理输出：\n{result.stdout}")
            except subprocess.CalledProcessError as e:
                QMessageBox.critical(None, "错误", f"运行失败：\n{e.stderr}")
            except FileNotFoundError:
                QMessageBox.critical(None, "错误", "未找到 Python 可执行文件，请确认环境设置")

        run_btn.clicked.connect(run_preprocessing)
        layout.addWidget(config)
        layout.addWidget(run_btn)
        layout.addStretch()
        return page

    def create_annotate_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        # ======= 参数自动化配置控件（ConfigFragment） =======
        config = ConfigFragment(
            "config.json",
            fields=[
                "annotate_img_dir",
                "pretrain_augment_times",
                "pretrain_device",
                "pretrain_model_name",
                "pretrain_batch_size",
                "pretrain_lr",
                "pretrain_epochs",
                "pretrain_warmup_factor",
                "pretrain_checkpoint_dir",
                "pretrain_checkpoint_filename",
                "pretrain_save_dir",
                "pretrain_save_filename",
                "annotate_dir"
            ],
            label_map={
                "annotate_img_dir":        "标注图像路径",
                "pretrain_augment_times":  "增强次数",
                "pretrain_device":         "训练设备",
                "pretrain_model_name":     "模型名称",
                "pretrain_batch_size":     "Batch Size",
                "pretrain_lr":             "学习率",
                "pretrain_epochs":         "训练轮数",
                "pretrain_warmup_factor":  "Warmup因子",
                "pretrain_checkpoint_dir": "Checkpoint目录",
                "pretrain_checkpoint_filename": "Checkpoint文件名",
                "pretrain_save_dir":       "模型保存目录",
                "pretrain_save_filename":  "模型文件名"
            }
        )

        # ======= 标注/训练/自动标注 按钮 =======
        start_labelme_btn = QPushButton("▶ 启动标注（labelme）")
        def start_labelme():
            with open("config.json", "r", encoding="utf-8") as f:
                cfg = json.load(f)
            folder = cfg.get("annotate_img_dir", "")
            if not folder:
                QMessageBox.warning(self, "未填写路径", "请填写标注图像路径")
                return
            try:
                #subprocess.Popen(["labelme", folder])
                subprocess.Popen(["python","run_labelme.py"],shell=True)
            except Exception as e:
                QMessageBox.critical(self, "启动失败", str(e))
        start_labelme_btn.clicked.connect(start_labelme)

        pretrain_btn = QPushButton("自动标注模型训练")
        def pretrain():
            try:
                launch_python_script("pretrain.py")
            except Exception as e:
                QMessageBox.critical(None, "错误", f"启动失败：{str(e)}")
        pretrain_btn.clicked.connect(pretrain)

        auto_annotate_btn = QPushButton("自动标注（TODO）")
        def auto_annotate():
            try:
                launch_python_script("auto_annotate.py")
            except Exception as e:
                QMessageBox.critical(None, "错误", f"启动失败：{str(e)}")
        auto_annotate_btn.clicked.connect(auto_annotate)

        # ======= 布局 =======
        layout.addWidget(config)
        layout.addWidget(start_labelme_btn)
        layout.addWidget(pretrain_btn)
        layout.addWidget(auto_annotate_btn)
        layout.addStretch()
        return page

    def create_train_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        config = ConfigFragment("config.json", fields=["model_name",
                                                       "in_channels",
                                                       "out_channels",
                                                       "input_size",
                                                       "num_classes",
                                                       "device",
                                                        "batch_size",
                                                        "accum_iter",
                                                        "use_amp",
                                                        "train_img_dir",
                                                        "train_mask_dir",
                                                        "val_img_dir",
                                                        "val_mask_dir",
                                                        "test_img_dir",
                                                        "test_mask_dir",
                                                        "save_dir",
                                                        "save_filename",
                                                        "checkpoint_dir",
                                                        "checkpoint_filename",
                                                        "log_dir",
                                                        "log_filename",
                                                        "lr", 
                                                        "epochs", 
                                                        "loss.use_ce",
                                                        "loss.use_bce",
                                                        "loss.use_dice",
                                                        "loss.use_focal",
                                                        "loss.use_boundary"],
                                label_map = {
                                                        "model_name": "模型名称",
                                                        "in_channels": "输入通道数",
                                                        "out_channels": "输出通道数",
                                                        "input_size": "输入图像大小",
                                                        "num_classes": "类别数量",
                                                        "device": "运行设备",
                                                        "batch_size": "批次大小",
                                                        "accum_iter": "梯度累积步数",
                                                        "use_amp": "使用混合精度 AMP",
                                                        "train_img_dir": "训练图像路径",
                                                        "train_mask_dir": "训练掩码路径",
                                                        "val_img_dir": "验证图像路径",
                                                        "val_mask_dir": "验证掩码路径",
                                                        "test_img_dir": "测试图像路径",
                                                        "test_mask_dir": "测试掩码路径",
                                                        "save_dir": "模型保存路径",
                                                        "checkpoint_dir": "检查点路径",
                                                        "log_dir": "日志 CSV 路径",
                                                        "lr": "学习率",
                                                        "epochs": "训练轮数",
                                                        "loss.use_ce": "使用交叉熵 (CE)",
                                                        "loss.use_bce": "使用二值交叉熵 (BCE)",
                                                        "loss.use_dice": "使用 Dice Loss",
                                                        "loss.use_focal": "使用 Focal Loss",
                                                        "loss.use_boundary": "使用 Boundary Loss"
                                                    }
                                                    )
        run_btn = QPushButton("▶ 开始训练")
        def run_in_cmd_window():
            try:
                launch_python_script("train.py")
            except Exception as e:
                QMessageBox.critical(None,"Mistake",f"Fail to launch:{str(e)}")
        run_btn.clicked.connect(run_in_cmd_window)        
        layout.addWidget(run_btn)
        layout.addStretch()
        return page

    def create_finetune_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        config = ConfigFragment("config.json", fields=[  "fine_tune_img_dir",
                                                       "fine_tune_mask_dir",
                                                       "fine_tune_epochs",
                                                       "fine_tune_lr",
                                                       "fine_tune_batch_size",
                                                       "fine_tune_save_dir",
                                                       "fine_tune_save_filename",
                                                       "freeze_encoder"])
        run_btn = QPushButton("▶ 开始微调")
        def run_finetune():
            try:
                launch_python_script("fine_tune.py")
            except Exception as e:
                QMessageBox.critical(None, "错误", f"启动失败：{str(e)}")
        run_btn.clicked.connect(run_finetune)
        layout.addWidget(config)
        layout.addWidget(run_btn)
        layout.addStretch()
        return page

    def create_predict_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        config = ConfigFragment("config.json", fields=[
                                                        "human_filter_dir",
                                                       "hum_filter_bad_picture_dir"])
        run_btn = QPushButton("▶ 开始推理")
        def run_predict():
            try:
                launch_python_script("human_filter.py")
            except Exception as e:
                QMessageBox.critical(None, "错误", f"启动失败：{str(e)}")
        run_btn.clicked.connect(run_predict)
        layout.addWidget(config)
        layout.addWidget(run_btn)
        layout.addStretch()
        return page
    
    def create_tcp_server_page(self):
        page=QWidget()
        layout=QVBoxLayout(page)
        config=ConfigFragment("config.json", fields=["model_name",
                                                     "in_channels",
                                                     "out_channels",
                                                     "input_size",
                                                     "save_dir",
                                                     "save_filename",
                                                     "device",
                                                     "host",
                                                     "port",
                                                     "max_threads"
                                                     ])
        run_btn= QPushButton("▶ 启动 TCP 服务器")
        def run_tcp_server():
            try:
                launch_python_script("tcp_server.py")
            except Exception as e:
                QMessageBox.critical(None, "错误", f"启动失败：{str(e)}")
        run_btn.clicked.connect(run_tcp_server)
        layout.addWidget(config)
        layout.addWidget(run_btn)
        layout.addStretch()
        return page


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainUI()
    win.show()
    sys.exit(app.exec_())
