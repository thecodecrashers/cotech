# run_labelme.py
import sys
import os
import json
import labelme.__main__

def main():
    with open("config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    folder = cfg.get("annotate_img_dir", "")
    if not folder or not os.path.exists(folder):
        print("❌ 路径未填写或不存在")
        return
    sys.argv = ["labelme", folder]
    labelme.__main__.main()

if __name__ == "__main__":
    main()
