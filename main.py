# D:\zhitu-treasure\check_dependencies.py
import sys
import importlib

# === 在这里列出你项目需要的所有模块 ===
REQUIRED_MODULES = [
    "torch",
    "torchvision",
    "torchaudio",
    "numpy",
    "PIL",          # Pillow
 
    "matplotlib",

    "tqdm",
    # 可根据实际需求增删
]

print("=" * 60)
print("🔍 依赖模块安装检查工具")
print(f"🐍 Python 路径: {sys.executable}")
print(f"📦 检查模块列表: {REQUIRED_MODULES}")
print("-" * 60)

all_passed = True

for module in REQUIRED_MODULES:
    try:
        if module == "PIL":
            # Pillow 的 import 名是 PIL
            importlib.import_module("PIL")

        else:
            importlib.import_module(module)
        print(f"✅ {module:<15} - 已安装")
    except ImportError as e:
        print(f"❌ {module:<15} - 未安装或导入失败: {e}")
        all_passed = False

print("-" * 60)
if all_passed:
    print("🎉 所有依赖模块均已正确安装！")
else:
    print("⚠️  部分模块缺失，请使用 pip 安装。")
    print("\n💡 安装建议命令（非 PyTorch 包可用清华源加速）：")
    print("pip install numpy opencv-python pillow matplotlib scikit-learn tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple")

print("=" * 60)