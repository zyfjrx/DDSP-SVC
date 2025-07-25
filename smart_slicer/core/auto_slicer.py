import os
import sys
import numpy as np
import librosa
import soundfile as sf

# 支持直接执行和模块导入两种方式
try:
    # 尝试相对导入（包模式）
    from .slicer import Slicer
except ImportError:
    # 如果相对导入失败，尝试直接导入（直接执行模式）
    try:
        from slicer import Slicer
    except ImportError:
        # 添加当前目录到路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from slicer import Slicer


class AutoSlicer:
    """
    智能音频切片器，提供自动音频切片、合并短音频等高级功能
    """
    
    def __init__(self):
        """初始化自动切片器"""
        self.slicer_params = {
            "threshold": -40,
            "min_length": 5000,
            "min_interval": 300,
            "hop_size": 10,
            "max_sil_kept": 500,
        }
        self.original_min_interval = self.slicer_params["min_interval"]

    def auto_slice(self, filename, input_dir, output_dir, max_sec):
        """
        自动切片单个音频文件
        
        Args:
            filename: 音频文件名
            input_dir: 输入目录
            output_dir: 输出目录
            max_sec: 最大切片秒数
        """
        audio, sr = librosa.load(os.path.join(input_dir, filename), sr=None, mono=False)
        slicer = Slicer(sr=sr, **self.slicer_params)
        chunks = slicer.slice(audio)
        files_to_delete = []
        
        for i, chunk in enumerate(chunks):
            if len(chunk.shape) > 1:
                chunk = chunk.T
                
            # 生成输出文件名（只保留ASCII字符和下划线）
            output_filename = f"{os.path.splitext(filename)[0]}_{i}"
            output_filename = "".join(c for c in output_filename if c.isascii() or c == "_") + ".wav"
            output_filepath = os.path.join(output_dir, output_filename)
            sf.write(output_filepath, chunk, sr)
            
            # 检查并重新切片超过最大时长的音频
            while True:
                new_audio, sr = librosa.load(output_filepath, sr=None, mono=False)
                if librosa.get_duration(y=new_audio, sr=sr) <= max_sec:
                    break
                    
                # 减少最小间隔参数以获得更细粒度的切片
                self.slicer_params["min_interval"] = self.slicer_params["min_interval"] // 2
                if self.slicer_params["min_interval"] >= self.slicer_params["hop_size"]:
                    new_chunks = Slicer(sr=sr, **self.slicer_params).slice(new_audio)
                    for j, new_chunk in enumerate(new_chunks):
                        if len(new_chunk.shape) > 1:
                            new_chunk = new_chunk.T
                        new_output_filename = f"{os.path.splitext(output_filename)[0]}_{j}.wav"
                        sf.write(os.path.join(output_dir, new_output_filename), new_chunk, sr)
                    files_to_delete.append(output_filepath)
                else:
                    break
                    
            # 恢复原始最小间隔参数
            self.slicer_params["min_interval"] = self.original_min_interval
            
        # 删除临时文件
        for file_path in files_to_delete:
            if os.path.exists(file_path):
                os.remove(file_path)

    def merge_short(self, output_dir, max_sec, min_sec):
        """
        将过短的音频片段合并为更长的音频
        
        Args:
            output_dir: 输出目录
            max_sec: 最大合并后时长
            min_sec: 最小音频时长阈值
        """
        short_files = []
        
        # 收集所有过短的音频文件
        for filename in os.listdir(output_dir):
            filepath = os.path.join(output_dir, filename)
            if filename.endswith(".wav"):
                audio, sr = librosa.load(filepath, sr=None, mono=False)
                duration = librosa.get_duration(y=audio, sr=sr)
                if duration < min_sec:
                    short_files.append((filepath, audio, duration))
        
        # 按时长倒序排列，优先处理较长的短音频
        short_files.sort(key=lambda x: x[2], reverse=True)
        merged_audio = []
        current_duration = 0
        
        for filepath, audio, duration in short_files:
            if current_duration + duration <= max_sec:
                # 可以继续添加到当前合并组
                merged_audio.append(audio)
                current_duration += duration
                os.remove(filepath)
            else:
                # 当前组已满，保存并开始新的合并组
                if merged_audio:
                    output_audio = np.concatenate(merged_audio, axis=-1)
                    if len(output_audio.shape) > 1:
                        output_audio = output_audio.T
                    output_filename = f"merged_{len(os.listdir(output_dir))}.wav"
                    sf.write(os.path.join(output_dir, output_filename), output_audio, sr)
                    
                # 开始新的合并组
                merged_audio = [audio]
                current_duration = duration
                os.remove(filepath)
        
        # 保存最后一组（如果满足最小时长要求）
        if merged_audio and current_duration >= min_sec:
            output_audio = np.concatenate(merged_audio, axis=-1)
            if len(output_audio.shape) > 1:
                output_audio = output_audio.T
            output_filename = f"merged_{len(os.listdir(output_dir))}.wav"
            sf.write(os.path.join(output_dir, output_filename), output_audio, sr)
    
    def slice_count(self, input_dir, output_dir):
        """
        统计切片结果
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            
        Returns:
            tuple: (文件数量, 最大时长, 最小时长, 原始总时长, 切片后总时长)
        """
        orig_duration = final_duration = 0
        
        # 计算原始音频总时长
        for file in os.listdir(input_dir):
            if file.endswith(".wav"):
                _audio, _sr = librosa.load(os.path.join(input_dir, file), sr=None, mono=False)
                orig_duration += librosa.get_duration(y=_audio, sr=_sr)
        
        # 统计切片后的音频信息
        wav_files = [file for file in os.listdir(output_dir) if file.endswith(".wav")]
        num_files = len(wav_files)
        max_duration = -1
        min_duration = float("inf")
        
        for file in wav_files:
            file_path = os.path.join(output_dir, file)
            audio, sr = librosa.load(file_path, sr=None, mono=False)
            duration = librosa.get_duration(y=audio, sr=sr)
            final_duration += float(duration)
            
            if duration > max_duration:
                max_duration = float(duration)
            if duration < min_duration:
                min_duration = float(duration)
                
        return num_files, max_duration, min_duration, orig_duration, final_duration 