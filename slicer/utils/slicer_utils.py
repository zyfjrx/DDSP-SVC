import os
import sys
import librosa
import shutil
from slicer.core.auto_slicer import AutoSlicer


def slice_audio_directory(input_dir, output_dir, process_method="丢弃", max_sec=15, min_sec=2, audio_format=".wav"):
    """
    智能音频切片函数，对目录中指定格式的音频文件进行切片处理
    
    Args:
        input_dir (str): 输入目录路径，包含待切片的音频文件
        output_dir (str): 输出目录路径，保存切片后的音频文件
        process_method (str): 对过短音频的处理方式，可选 "丢弃" 或 "将过短音频整合为长音频"
        max_sec (float): 最大切片时长（秒）
        min_sec (float): 最小切片时长（秒）
        audio_format (str): 音频格式，如 ".wav", ".mp3" 等
        
    Returns:
        str: 处理结果报告
        
    Raises:
        ValueError: 当参数不合法时抛出异常
    """
    # 参数验证
    if output_dir == "":
        raise ValueError("请先选择输出的文件夹")
    if output_dir == input_dir:
        raise ValueError("输出目录不能和输入目录相同")
    if not os.path.exists(input_dir):
        raise ValueError(f"输入目录不存在: {input_dir}")
    if max_sec <= min_sec:
        raise ValueError("最大切片时长必须大于最小切片时长")
        
    # 创建输出目录（如果不存在）或清空现有目录
    if os.path.exists(output_dir):
        # 如果输出目录存在，先清空其中的所有文件
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"删除文件 {file_path} 时出错: {e}")
        print(f"已清空输出目录: {output_dir}")
    else:
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")
    
    # 初始化自动切片器
    slicer = AutoSlicer()
    
    # 处理输入目录中指定格式的音频文件
    processed_files = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(audio_format.lower()):
            try:
                slicer.auto_slice(filename, input_dir, output_dir, max_sec)
                processed_files += 1
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
                continue
    
    if processed_files == 0:
        return f"未找到可处理的 {audio_format} 音频文件"
    
    # 根据处理方式处理过短音频
    if process_method == "丢弃":
        # 删除过短的音频文件
        removed_count = 0
        for filename in os.listdir(output_dir):
            if filename.endswith(".wav"):
                filepath = os.path.join(output_dir, filename)
                try:
                    audio, sr = librosa.load(filepath, sr=None, mono=False)
                    if librosa.get_duration(y=audio, sr=sr) < min_sec:
                        os.remove(filepath)
                        removed_count += 1
                except Exception as e:
                    print(f"检查文件 {filename} 时出错: {e}")
                    continue
        if removed_count > 0:
            print(f"删除了 {removed_count} 个过短的音频文件")
            
    elif process_method == "将过短音频整合为长音频":
        # 合并过短的音频文件
        try:
            slicer.merge_short(output_dir, max_sec, min_sec)
        except Exception as e:
            print(f"合并短音频时出错: {e}")
    
    # 统计切片结果
    try:
        file_count, max_duration, min_duration, orig_duration, final_duration = slicer.slice_count(input_dir, output_dir)
        
        # 格式化时间显示
        hrs = int(final_duration / 3600)
        mins = int((final_duration % 3600) / 60)
        sec = format(float(final_duration % 60), '.2f')
        
        # 计算时长占比
        rate = format(100 * (final_duration / orig_duration), '.2f') if orig_duration != 0 else 0
        rate_msg = f"为原始音频时长的{rate}%" if rate != 0 else "因未知问题，无法计算切片时长的占比"
        
        return (f"成功处理了 {processed_files} 个音频文件，"
                f"切分为 {file_count} 条片段，"
                f"其中最长 {max_duration:.2f} 秒，最短 {min_duration:.2f} 秒，"
                f"切片后的音频总时长 {hrs:02d}小时{mins:02d}分{sec}秒，{rate_msg}")
                
    except Exception as e:
        return f"切片完成，但统计结果时出错: {e}"


def validate_audio_directory(directory_path, audio_format=".wav"):
    """
    验证音频目录是否包含指定格式的音频文件
    
    Args:
        directory_path (str): 目录路径
        audio_format (str): 音频格式，如 ".wav", ".mp3" 等
        
    Returns:
        tuple: (是否有效, 错误信息或音频文件列表)
    """
    if not os.path.isdir(directory_path):
        return False, "请输入正确的目录"
    
    files = os.listdir(directory_path)
    audio_files = [file for file in files if file.lower().endswith(audio_format.lower())]
    
    if not audio_files:
        return False, f"未在目录中找到 {audio_format} 音频文件"
    
    return True, audio_files