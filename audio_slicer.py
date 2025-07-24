import os
import sys
import traceback
import subprocess
import gradio as gr

# 直接导入需要的模块
from slicer.core.slicer import Slicer, get_rms
from slicer.core.auto_slicer import AutoSlicer
from slicer.utils.slicer_utils import slice_audio_directory, validate_audio_directory

def check_and_install_dependencies():
    """检查并安装依赖"""
    print("🔍 检查依赖包...")

    # 核心依赖包
    dependencies = {
        'numpy': 'numpy>=1.21.0',
        'librosa': 'librosa>=0.9.0',
        'soundfile': 'soundfile>=0.10.0',
        'gradio': 'gradio>=4.0.0'
    }

    missing = []
    for package, requirement in dependencies.items():
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"❌ {package} 未安装")
            missing.append(requirement)

    if missing:
        print(f"\n📦 正在安装缺失依赖: {', '.join(missing)}")
        try:
            cmd = [sys.executable, "-m", "pip", "install"] + missing
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ 依赖安装完成")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ 安装失败: {e}")
            print("请手动运行: pip install " + " ".join(missing))
            return False
    return True


def load_raw_audio(audio_path, audio_format=".wav"):
    """加载指定格式的音频文件列表"""
    if not audio_path.strip():
        return "请输入音频目录路径", None

    is_valid, result = validate_audio_directory(audio_path, audio_format)

    if not is_valid:
        return result, None
    else:
        audio_files = result
        file_list_text = "\n".join(audio_files[:10])
        if len(audio_files) > 10:
            file_list_text += f"\n... 还有 {len(audio_files) - 10} 个文件"
        return f"成功加载 {len(audio_files)} 个 {audio_format} 音频文件", file_list_text


def slice_audio_ui(input_dir, output_dir, process_method, max_sec, min_sec, audio_format=".wav", progress=gr.Progress()):
    """UI 切片处理函数"""
    try:
        if not input_dir.strip():
            return "❌ 请先输入原始音频文件夹路径"

        if not output_dir.strip():
            return "❌ 请先输入输出目录路径"

        progress(0, desc="准备开始音频切片...")

        result = slice_audio_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            process_method=process_method,
            max_sec=max_sec,
            min_sec=min_sec,
            audio_format=audio_format
        )

        progress(1, desc="切片完成!")
        return f"✅ {result}"

    except Exception as e:
        error_msg = f"❌ 切片过程中出现错误: {str(e)}"
        print(f"Error: {e}")
        traceback.print_exc()
        return error_msg