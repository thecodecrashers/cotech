# utils.py
import sys
import os
import subprocess
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
