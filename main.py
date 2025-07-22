import sys
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

        for btn in [self.btn_annotate, self.btn_data, self.btn_train, self.btn_predict,self.btn_finetune,]:
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

        self.stack.addWidget(self.page_annotate)
        self.stack.addWidget(self.page_data)
        self.stack.addWidget(self.page_train)
        self.stack.addWidget(self.page_predict)
        self.stack.addWidget(self.page_finetune)
        # 按钮绑定
        self.btn_data.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_data))
        self.btn_annotate.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_annotate))
        self.btn_train.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_train))
        self.btn_finetune.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_finetune))
        self.btn_predict.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_predict))

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
                # 路径根据实际放置的位置修改
                result=subprocess.run(["python", "utils/split_labeled_dataset.py"], check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True)
                QMessageBox.information(self, "完成", f"处理输出：\n{result.stdout}")
            except subprocess.CalledProcessError as e:
                QMessageBox.critical(self, "错误", f"运行失败：\n{e}")
            except FileNotFoundError:
                QMessageBox.critical(self, "错误", "未找到 Python 可执行文件，请确认环境设置")

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
                "pretrain_save_filename"
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
            import subprocess
            import json
            with open("config.json", "r", encoding="utf-8") as f:
                cfg = json.load(f)
            folder = cfg.get("annotate_img_dir", "")
            if not folder:
                QMessageBox.warning(self, "未填写路径", "请填写标注图像路径")
                return
            try:
                subprocess.Popen(["labelme", folder])
            except Exception as e:
                QMessageBox.critical(self, "启动失败", str(e))
        start_labelme_btn.clicked.connect(start_labelme)

        pretrain_btn = QPushButton("自动标注模型训练")
        def pretrain():
            try:
                if platform.system() == "Windows":
                    subprocess.Popen(["start", "cmd", "/k", "python pretrain.py"], shell=True)
                else:
                    # Linux/macOS 示例，使用 gnome-terminal / bash
                    subprocess.Popen(["x-terminal-emulator", "-e", "python3 train.py"])
            except Exception as e:
                QMessageBox.critical(None, "错误", f"启动失败：{str(e)}")
        # def pretrain():
        #     # 路径根据实际情况填写
        #     try:
        #         result = subprocess.run(["python", "pretrain.py"], check=True,
        #             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        #         QMessageBox.information(self, "训练完成", result.stdout[-1000:])
        #     except subprocess.CalledProcessError as e:
        #         QMessageBox.critical(self, "训练失败", e.stderr)
        pretrain_btn.clicked.connect(pretrain)

        auto_annotate_btn = QPushButton("自动标注（TODO）")
        def auto_annotate():
            try:
                if platform.system() == "Windows":
                    subprocess.Popen(["start", "cmd", "/k", "python auto_annotate.py"], shell=True)
                else:
                    subprocess.Popen(["x-terminal-emulator", "-e", "python3 auto_annotate.py"])
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


#region
    # def create_annotate_page(self):
    #     page = QWidget()
    #     layout = QVBoxLayout(page)

    #     # ===== 图像路径输入区 =====
    #     path_layout = QHBoxLayout()
    #     label = QLabel("📂 原始图像路径：")
    #     label.setFixedWidth(150)

    #     with open("config.json", "r", encoding="utf-8") as f:
    #         cfg = json.load(f)
    #     default_path = cfg.get("annotate_img_dir", "")

    #     self.annotate_img_input = QLineEdit(default_path)
    #     self.annotate_img_input.setMinimumWidth(400)

    #     browse_btn = QPushButton("📁 浏览")
    #     def browse_folder():
    #         folder = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
    #         if folder:
    #             self.annotate_img_input.setText(folder)
    #     browse_btn.clicked.connect(browse_folder)  

    #     path_layout.addWidget(label)
    #     path_layout.addWidget(self.annotate_img_input)
    #     path_layout.addWidget(browse_btn)
    #     layout.addLayout(path_layout)

    #     # 保存路径按钮
    #     save_btn = QPushButton("💾 保存路径到 config.json")
    #     def save_path():
    #         folder = self.annotate_img_input.text().strip()
    #         if not folder:
    #             QMessageBox.warning(self, "路径为空", "请输入或选择一个文件夹路径")
    #             return
    #         try:
    #             with open("config.json", "r", encoding="utf-8") as f:
    #                 cfg = json.load(f)
    #             cfg["annotate_img_dir"] = folder
    #             with open("config.json", "w", encoding="utf-8") as f:
    #                 json.dump(cfg, f, indent=2, ensure_ascii=False)
    #             QMessageBox.information(self, "成功", "路径已保存到 config.json")
    #         except Exception as e:
    #             QMessageBox.critical(self, "保存失败", str(e))
    #     save_btn.clicked.connect(save_path)
    #     layout.addWidget(save_btn)

    #     # 启动标注按钮
    #     start_btn = QPushButton("▶ 启动标注（打开 labelme）")
    #     def start_labelme():
    #         folder = self.annotate_img_input.text().strip()
    #         if not folder:
    #             QMessageBox.warning(self, "路径未填写", "请先选择原始图像路径")
    #             return
    #         try:
    #             subprocess.Popen(["labelme", folder])
    #         except FileNotFoundError:
    #             QMessageBox.critical(self, "未找到 labelme", "请确保已安装 labelme 并添加到环境变量中。")
                
    #     pretrain_btn=QPushButton("自动标注模型训练")
    #     auto_annotate_btn=QPushButton("自动标注（TODO）")
    #     start_btn.clicked.connect(start_labelme)
    #     layout.addWidget(start_btn)
    #     layout.addWidget(pretrain_btn)
    #     layout.addWidget(auto_annotate_btn)
    #     layout.addStretch()
    #     return page
#endregion

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
                if platform.system() == "Windows":
                    subprocess.Popen(["start", "cmd", "/k", "python train.py"], shell=True)
                else:
                    # Linux/macOS 示例，使用 gnome-terminal / bash
                    subprocess.Popen(["x-terminal-emulator", "-e", "python3 train.py"])
            except Exception as e:
                QMessageBox.critical(None, "错误", f"启动失败：{str(e)}")

        run_btn.clicked.connect(run_in_cmd_window)        
        # def run_preprocessing():
        #     try:
        #         result = subprocess.run(
        #             ["python", "train.py"],
        #             check=True,
        #             stdout=subprocess.PIPE,
        #             stderr=subprocess.PIPE,
        #             text=True
        #         )
        #         QMessageBox.information(None, "完成", f"输出：\n{result.stdout}")
        #     except subprocess.CalledProcessError as e:
        #         # 打印完整错误信息包括 stderr
        #         QMessageBox.critical(None, "运行失败", f"错误代码：{e.returncode}\n\nstderr:\n{e.stderr}")
        #     except FileNotFoundError:
        #         QMessageBox.critical(None, "错误", "找不到 Python，请确认环境路径")

        # run_btn.clicked.connect(run_preprocessing)        
        layout.addWidget(config)
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
                                                       "fine_tune_save_path",
                                                       "freeze_encoder"])
        run_btn = QPushButton("▶ 开始微调")
        layout.addWidget(config)
        layout.addWidget(run_btn)
        layout.addStretch()
        return page

    def create_predict_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        config = ConfigFragment("config.json", fields=[
                                                        "human_filter_path",
                                                       "hum_filter_bad_picture"])
        run_btn = QPushButton("▶ 开始推理")
        def run_predict():
            try:
                if platform.system() == "Windows":
                    subprocess.Popen(["start", "cmd", "/k", "python human_filter.py"], shell=True)
                else:
                    # Linux/macOS 示例，使用 gnome-terminal / bash
                    subprocess.Popen(["x-terminal-emulator", "-e", "python3 human_filter.py"])
            except Exception as e:
                QMessageBox.critical(None, "错误", f"启动失败：{str(e)}")
        run_btn.clicked.connect(run_predict)
        layout.addWidget(config)
        layout.addWidget(run_btn)
        layout.addStretch()
        return page


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainUI()
    win.show()
    sys.exit(app.exec_())






# import sys
# import json
# from PyQt5.QtWidgets import (
#     QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
#     QLineEdit, QPushButton, QMessageBox, QFileDialog,
#     QStackedLayout
# )
# from PyQt5.QtCore import Qt


# # ========================== 配置编辑器组件 ==========================

# class ConfigEditor(QWidget):
#     def __init__(self, config_path="config.json"):
#         super().__init__()
#         self.setWindowTitle("配置编辑器")
#         self.config_path = config_path
#         self.inputs = {}

#         self.layout = QVBoxLayout(self)
#         self.load_config()
#         self.build_ui()
#         self.add_buttons()

#     def load_config(self):
#         with open(self.config_path, "r", encoding="utf-8") as f:
#             self.config = json.load(f)

#     def add_row(self, label_text, key_path, browse=False, is_file=False):
#         hbox = QHBoxLayout()

#         label = QLabel(label_text)
#         label.setFixedWidth(200)

#         value = self.get_value_by_path(key_path)
#         edit = QLineEdit(str(value))
#         edit.setMinimumWidth(300)

#         hbox.addWidget(label)
#         hbox.addWidget(edit)

#         if browse:
#             btn = QPushButton("📂")
#             btn.setFixedWidth(40)

#             def open_dialog():
#                 if is_file:
#                     path, _ = QFileDialog.getSaveFileName(self, "选择文件")
#                 else:
#                     path = QFileDialog.getExistingDirectory(self, "选择文件夹")
#                 if path:
#                     edit.setText(path)

#             btn.clicked.connect(open_dialog)
#             hbox.addWidget(btn)

#         self.layout.addLayout(hbox)
#         self.inputs[key_path] = edit

#     def get_value_by_path(self, path):
#         keys = path.split('.')
#         value = self.config
#         for k in keys:
#             value = value[k]
#         return value

#     def set_value_by_path(self, path, new_value):
#         keys = path.split('.')
#         obj = self.config
#         for k in keys[:-1]:
#             obj = obj[k]
#         obj[keys[-1]] = new_value

#     def build_ui(self):
#         self.add_row("模型名", "model_name")
#         self.add_row("输入通道数", "in_channels")
#         self.add_row("输出通道数", "out_channels")
#         self.add_row("是否使用 AMP", "use_amp")
#         self.add_row("训练图像路径", "train_img_dir", browse=True)
#         self.add_row("训练掩码路径", "train_mask_dir", browse=True)
#         self.add_row("学习率", "lr")
#         self.add_row("训练轮数", "epochs")
#         self.add_row("交叉熵损失", "loss.use_ce")
#         self.add_row("Dice 损失", "loss.use_dice")

#     def add_buttons(self):
#         btn_layout = QHBoxLayout()

#         save_btn = QPushButton("💾 保存配置")
#         save_btn.clicked.connect(self.save_config)

#         reload_btn = QPushButton("🔁 重新加载")
#         reload_btn.clicked.connect(self.reload_config)

#         btn_layout.addWidget(save_btn)
#         btn_layout.addWidget(reload_btn)
#         self.layout.addLayout(btn_layout)

#     def save_config(self):
#         for key_path, edit in self.inputs.items():
#             raw = edit.text()
#             try:
#                 val = eval(raw, {}, {})
#             except:
#                 val = raw
#             self.set_value_by_path(key_path, val)

#         with open(self.config_path, "w", encoding="utf-8") as f:
#             json.dump(self.config, f, indent=2, ensure_ascii=False)

#         QMessageBox.information(self, "成功", "配置保存成功！")

#     def reload_config(self):
#         self.inputs.clear()
#         self.load_config()
#         QMessageBox.information(self, "重新加载", "配置已重新加载")



# # ========================== 主界面 ==========================

# class MainUI(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("工业视觉训练平台")
#         self.resize(900, 650)

#         self.layout = QVBoxLayout(self)

#         # 三个主功能按钮
#         self.btn_config = QPushButton("🧩 配置管理")
#         self.btn_train = QPushButton("🧠 模型训练")
#         self.btn_predict = QPushButton("📤 模型推理")

#         for btn in [self.btn_config, self.btn_train, self.btn_predict]:
#             btn.setFixedHeight(40)
#             self.layout.addWidget(btn)

#         # 页面切换容器
#         self.stack = QStackedLayout()
#         self.layout.addLayout(self.stack)

#         # 每个页面内容
#         self.page_config = ConfigEditor("config.json")

#         self.page_train = QLabel("👉 这里是训练页面（TODO）")
#         self.page_train.setAlignment(Qt.AlignCenter)

#         self.page_predict = QLabel("👉 这里是推理页面（TODO）")
#         self.page_predict.setAlignment(Qt.AlignCenter)

#         # 添加页面
#         self.stack.addWidget(self.page_config)
#         self.stack.addWidget(self.page_train)
#         self.stack.addWidget(self.page_predict)

#         # 按钮绑定页面切换
#         self.btn_config.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_config))
#         self.btn_train.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_train))
#         self.btn_predict.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_predict))


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     win = MainUI()
#     win.show()
#     sys.exit(app.exec_())   