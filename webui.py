import gradio as gr
import os
import subprocess
import yaml
import threading
import time
import json
import shutil
from typing import Optional, Tuple, List
import sys
import datetime
import matplotlib.pyplot as plt
import matplotlib
import glob
from ml_collections import ConfigDict
from tools.smart_slicer.audio_slicer import slice_audio_ui, load_raw_audio
from tools.SOME.infer import some_inference
# 添加 MSST 相关导入
import warnings
import logging
sys.path.append('tools/msst')
from tools.msst.inference.msst_infer import MSSeparator
from tools.msst.utils.logger import get_logger


class Info:
    def __init__(self):
        self.preprocess = """
### 数据预处理说明

**数据组织功能:**
1. **原始数据处理**: 从指定目录读取已切分好的音频文件
2. **自动格式转换**: 支持多种音频格式(wav/mp3/flac等)
3. **数据分配**: 自动分配训练集和验证集
4. **目录组织**: 按照README规则自动组织数据结构
5. **音频标准化**: 重采样和音量标准化处理

**单人模型结构:**
```
data/train/audio/train_0001.wav
data/val/audio/val_0001.wav
```

**多人模型结构:**
```
data/train/audio/1/spk1_train_0001.wav
data/train/audio/2/spk2_train_0001.wav
data/val/audio/1/spk1_val_0001.wav
data/val/audio/2/spk2_val_0001.wav
```

**特征提取步骤:**
1. **F0提取**: 提取音频的基频信息
2. **音量提取**: 计算音频的音量包络
3. **梅尔频谱**: 提取梅尔频谱特征
4. **单位特征**: 使用编码器提取语义特征

**使用建议:**
- 确保原始音频已经切分为2-30秒的短音频
- 音频质量要求: 清晰无噪音，采样率一致
- 先使用"组织数据"整理音频文件
- 单人模型建议1000个左右音频文件
- 多人模型每个说话人建议500-1000个文件
- 验证集保持10-30个文件即可
        """
        
        self.train = """
### 模型训练说明

**训练流程:**
1. 加载预处理的特征文件
2. 初始化模型和优化器
3. 开始训练循环
4. 定期保存检查点
5. 生成训练日志

**训练参数:**
- **Epoch**: 训练轮次，建议100-500轮
- **批次大小**: 根据显存调整，通常8-32
- **学习率**: 建议0.0001-0.001
- **保存间隔**: 每10-20轮保存一次

**监控指标:**
- 损失值 (Loss)
- 学习率变化
- 训练时间

**注意事项:**
- 训练过程中可随时停止
- 建议使用TensorBoard监控训练过程
- 定期检查生成的音频质量
        """
        
        self.infer = """
### 音频推理说明

**推理流程:**
1. 加载训练好的模型
2. 上传待转换的音频文件
3. 设置转换参数
4. 执行音频转换（实时显示进度）
5. 下载转换结果

**参数说明:**
- **变调**: 音高调整，单位为半音
- **说话人ID**: 多说话人模型时选择目标说话人
- **推理步数**: 影响质量和速度，建议30-100步

**进度监控:**
- **整体进度**: 显示转换的总体完成百分比
- **当前状态**: 显示关键处理节点（初始化、加载模型、音频处理、保存结果、完成）
- **简化日志**: 显示关键处理信息，避免冗余输出

**音频要求:**
- 支持WAV、MP3等常见格式
- 建议采样率与训练时一致
- 时长建议在30秒以内

**输出结果:**
- 转换后的音频文件
- 保存在results目录下
- 可直接播放和下载

**进度阶段说明:**
- 0-10%: 系统初始化和准备
- 10-25%: 模型加载
- 25-85%: 音频推理处理
- 85-95%: 结果保存
- 95-100%: 处理完成
        """

class EnhancedWebUI:
    def __init__(self):
        self.info = Info()
        self.opt_cfg_pth = 'configs/reflow.yaml'
        self.current_process = None
        self.log_content = ""
        self.training_status = "未开始"
        self.preprocessing_status = "未开始"
        self.training_log_cache = ""
        self.current_epoch = 0
        self.current_step = 0
        self.loss_history = []
        self.auto_refresh_active = False
        self.auto_refresh_thread = None
        # 添加 MSST 相关状态
        self.msst_process = None
        self.msst_status = "未开始"
        self.msst_pretrain_dir = "tools/msst/pretrain"
        self.current_config_data = None  # 存储当前配置数据
        self.ensure_directories()
    
    def ensure_directories(self):
        """确保必要的目录存在"""
        dirs = [
            'data/train/audio',
            'data/val/audio', 
            'results',
            'exp',
            'pretrain/contentvec',
            'pretrain/nsf_hifigan'
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def create_interface(self):
        """创建 Gradio 界面"""
        with gr.Blocks(title="DDSP-SVC WebUI", theme=gr.themes.Soft()) as ui:
            gr.Markdown("# 🎵 DDSP-SVC WebUI")
            gr.Markdown("一个便于训练和推理的DDSP-SVC界面，支持数据预处理、模型训练和音频转换")
            
            # 添加 MSST 音频分离标签页
            with gr.Tab("🎼 音频分离 (MSST)"):
                self.create_msst_tab()

            with gr.Tab("🔪 智能音频切分"):
                self.audio_slicer()

            with gr.Tab("🔄 数据预处理"):
                self.create_preprocessing_tab()
            
            with gr.Tab("🚀 模型训练"):
                self.create_training_tab()
            
            with gr.Tab("🎤 音频推理"):
                self.create_inference_tab()

            with gr.Tab("歌声转MIDI"):
                self.exp_midi()
            
            # 添加 MSST 音频分离标签页
            # with gr.Tab("🎼 音频分离 (MSST)"):
            #     self.create_msst_tab()
            
            with gr.Tab("📊 监控面板"):
                self.create_monitoring_tab()
        
        return ui
    
    def exp_midi(self):
        with gr.Row():
            some_input_audio = gr.Audio(label="上传音频", type="filepath")
            with gr.Column():
            	audio_bpm = gr.Number(label="输入音频BPM", value=120, interactive=True)
            	some_output_folder = gr.Textbox(label="输出目录", value="results/mid", interactive=True, scale=3)
        with gr.Row():
            some_button = gr.Button("开始转换", variant="primary")
            output_message_some = gr.Textbox(label="Output Message")
        some_button.click(fn=some_inference, inputs=[some_input_audio, audio_bpm, some_output_folder], outputs=output_message_some)



    def audio_slicer(self):
        """智能音频切分UI界面"""

        gr.HTML('<h3 style="text-align: center;">🎵 智能音频切分工具</h3>')
        gr.HTML('<p style="text-align: center; color: #666;">基于静音检测的音频自动切片，支持批量处理</p>')

        with gr.Row():
            raw_audio_path = gr.Textbox(
                label="原始音频文件夹",
                placeholder="输入包含音频文件的目录路径",
                value="input/",
                scale=4
            )
            audio_format = gr.Dropdown(
                choices=[".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"],
                value=".wav",
                label="🎵 音频格式",
                scale=2
            )
            load_raw_audio_btn = gr.Button("🔍 加载原始音频", variant="primary")

        with gr.Row():
            
            load_raw_audio_output = gr.Textbox(
                label="📋 加载状态",
                interactive=False,
                max_lines=3
            )

            raw_audio_dataset = gr.Textbox(
                label="🎵 音频文件列表",
                interactive=False,
                max_lines=8,
                placeholder="点击'加载原始音频'后显示文件列表"
            )
        # with gr.Row():
        #     slicer_output_dir = gr.Textbox(
        #         label="输出目录",
        #         placeholder="输入保存切片结果的目录路径（不要和输入目录相同）",
        #         value="results/",
        #         scale=4
        #     )

        with gr.Row():
            with gr.Column():
                process_method = gr.Radio(
                    label="过短音频处理方式",
                    choices=["丢弃", "将过短音频整合为长音频"],
                    value="丢弃"
                )
                slicer_output_dir = gr.Textbox(
                    label="输出目录",
                    placeholder="输入保存切片结果的目录路径（不要和输入目录相同）",
                    value="results/",
                    scale=4
                )
            with gr.Column():
                max_sec = gr.Number(
                    label="最大切片时长（秒）",
                    value=15,
                    minimum=1
                )
                min_sec = gr.Number(
                    label="最小切片时长（秒）",
                    value=2,
                    minimum=1
                )

        slicer_btn = gr.Button("🎯 开始切片", variant="primary", size="lg")
        slicer_output_msg = gr.Textbox(
            label="🔄 切片结果",
            interactive=False,
            max_lines=5,
            placeholder="点击'开始切片'后显示结果"
        )

        # 事件绑定 - 修改为传递音频格式参数
        load_raw_audio_btn.click(
            fn=load_raw_audio,
            inputs=[raw_audio_path, audio_format],
            outputs=[load_raw_audio_output, raw_audio_dataset]
        )

        slicer_btn.click(
            fn=slice_audio_ui,
            inputs=[raw_audio_path, slicer_output_dir, process_method, max_sec, min_sec, audio_format],
            outputs=[slicer_output_msg]
        )

        # ... existing code ...
    def create_preprocessing_tab(self):
        """数据预处理标签页"""
        gr.Markdown("## 🔄 数据预处理")
        
        with gr.Accordion("预处理说明", open=False):
            gr.Markdown(self.info.preprocess)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📁 数据源设置")
                source_data_dir = gr.Textbox(
                    label="原始数据目录",
                    placeholder="例如: /root/my_audio_data",
                    info="包含所有原始音频文件的目录"
                )
                
                model_type = gr.Radio(
                    choices=["single", "multi"],
                    value="single",
                    label="模型类型",
                    info="single: 单人模型, multi: 多人模型"
                )
                
                # 单人模型设置
                with gr.Group(visible=True) as single_speaker_group:
                    gr.Markdown("单人模型设置")
                    train_ratio = gr.Slider(
                        minimum=0.6, maximum=0.95, value=0.9, step=0.05,
                        label="训练集比例",
                        info="剩余部分将作为验证集"
                    )
                
                # 多人模型设置
                with gr.Group(visible=False) as multi_speaker_group:
                    gr.Markdown("多人模型设置")
                    speaker_mapping = gr.Textbox(
                        label="说话人映射",
                        placeholder='{"speaker1": 1, "speaker2": 2}',
                        info="JSON格式，将文件夹名映射到说话人ID",
                        lines=3
                    )
                    multi_train_ratio = gr.Slider(
                        minimum=0.6, maximum=0.95, value=0.9, step=0.05,
                        label="训练集比例",
                        info="每个说话人的训练集比例"
                    )
            
            with gr.Column():
                gr.Markdown("### ⚙️ 预处理选项")
                
                audio_format = gr.Dropdown(
                    choices=[".wav", ".mp3", ".flac", ".m4a", ".aac"],
                    value=".wav",
                    label="音频格式",
                    info="要处理的音频文件格式"
                )
                
                target_sr = gr.Number(
                    value=44100, minimum=16000, maximum=48000,
                    label="目标采样率",
                    info="所有音频将重采样到此采样率"
                )
                
                normalize_audio = gr.Checkbox(
                    value=True,
                    label="音频标准化",
                    info="标准化音频音量"
                )
        
        with gr.Row():
            organize_btn = gr.Button("📋 组织数据", variant="secondary")
            preprocess_btn = gr.Button("🚀 开始预处理", variant="primary")
            stop_preprocess_btn = gr.Button("⏹️ 停止预处理", variant="stop")
        
        with gr.Row():
            with gr.Column():
                preprocess_status = gr.Textbox(
                    label="预处理状态",
                    value="未开始",
                    interactive=False
                )
                preprocess_progress = gr.Slider(
                    minimum=0, maximum=100, value=0, step=1,
                    label="预处理进度 (%)",
                    interactive=False
                )
            with gr.Column():
                data_summary = gr.Textbox(
                    label="数据统计",
                    value="",
                    interactive=False
                )
        
        # 移除了预处理日志输出框
        # preprocess_log = gr.Textbox(
        #     label="预处理日志",
        #     lines=12,
        #     interactive=False,
        #     autoscroll=True
        # )
        
        # 模型类型切换事件
        def toggle_speaker_groups(model_type):
            if model_type == "single":
                return gr.Group(visible=True), gr.Group(visible=False)
            else:
                return gr.Group(visible=False), gr.Group(visible=True)
        
        model_type.change(
            fn=toggle_speaker_groups,
            inputs=[model_type],
            outputs=[single_speaker_group, multi_speaker_group]
        )
        
        # 绑定事件 - 移除了preprocess_log输出
        organize_btn.click(
            fn=self.organize_data,
            inputs=[source_data_dir, model_type, train_ratio, multi_train_ratio, 
                   speaker_mapping, audio_format, target_sr, normalize_audio],
            outputs=[preprocess_status, data_summary]  # 移除了preprocess_log
        )
        
        preprocess_btn.click(
            fn=self.start_preprocessing_with_progress,
            outputs=[preprocess_status, preprocess_progress],  # 移除了preprocess_log
            show_progress=True
        )
        
        stop_preprocess_btn.click(
            fn=self.stop_preprocessing,
            outputs=[preprocess_status, preprocess_progress]
        )
    
    def create_training_tab(self):
        """模型训练标签页"""
        gr.Markdown("## 🚀 模型训练")
        
        with gr.Accordion("训练说明", open=False):
            gr.Markdown(self.info.train)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 基础训练参数")
                batch_size = gr.Number(label="Batch Size", value=24, precision=0, minimum=1)
                learning_rate = gr.Number(label="Learning Rate", value=0.0005, minimum=0.0001)
                total_epochs = gr.Number(label="训练轮数", value=100000, precision=0, minimum=1)
                
            with gr.Column():
                gr.Markdown("### 检查点设置")
                interval_val = gr.Number(label="验证间隔(步数)", value=2000, precision=0, minimum=100)
                interval_force_save = gr.Number(label="强制保存间隔(步数)", value=10000, precision=0, minimum=1000)
                interval_log = gr.Number(label="日志打印间隔(步数)", value=1, precision=0, minimum=1)
        
        with gr.Row():
            start_btn = gr.Button("🚀 开始训练", variant="primary")
            stop_btn = gr.Button("⏹️ 停止训练", variant="stop")
        
        with gr.Row():
            training_status = gr.Textbox(label="训练状态", value="💤 未开始训练")
            current_epoch = gr.Textbox(label="当前轮数", value="0")
            current_step = gr.Textbox(label="当前步数", value="0")
        
        loss_plot = gr.Plot(label="训练Loss曲线")
        
        with gr.Row():
            refresh_btn = gr.Button("🔄 刷新训练信息", variant="secondary")
            force_stop_btn = gr.Button("🆘 强制停止", variant="stop")
        
        gr.Markdown("💡 **使用说明**: 训练启动后可点击 '🔄 刷新训练信息' 查看最新进度。")
        
        # 绑定事件
        start_btn.click(
            fn=self.start_training,
            inputs=[batch_size, learning_rate, total_epochs, interval_val, interval_force_save, interval_log],
            outputs=[training_status, current_epoch, current_step]
        )
        
        stop_btn.click(
            fn=self.stop_training,
            outputs=[training_status, current_epoch, current_step]
        )
        
        refresh_btn.click(
            fn=self.get_training_info,
            outputs=[training_status, current_epoch, current_step, loss_plot]
        )
        
        force_stop_btn.click(
            fn=self.force_stop_training,
            outputs=[training_status, current_epoch, current_step]
        )
    
    def create_inference_tab(self):
        """音频推理标签页"""
        gr.Markdown("## 🎤 音频推理")
        
        with gr.Accordion("推理说明", open=False):
            gr.Markdown(self.info.infer)
        
        with gr.Row():
            with gr.Column():
                input_audio = gr.Audio(
                    label="输入音频",
                    type="filepath"
                )
                
                # 获取可用的时间戳目录
                timestamps = self.get_model_timestamps()
                initial_models = self.get_model_files(timestamps[0] if timestamps else None)
                
                # 添加模型选择下拉框
                with gr.Row():
                    timestamp_dropdown = gr.Dropdown(
                        choices=timestamps,
                        value=timestamps[0] if timestamps else None,
                        label="训练时间",
                        info="选择训练时间戳文件夹",
                        interactive=True
                    )
                    model_dropdown = gr.Dropdown(
                        choices=initial_models,
                        value=initial_models[0] if initial_models else None,
                        label="模型文件",
                        info="选择模型检查点文件",
                        interactive=True
                    )
                
                refresh_btn = gr.Button("🔄 刷新模型列表", variant="secondary")
            
            with gr.Column():
                key_change = gr.Slider(
                    minimum=-24, maximum=24, value=0, step=1,
                    label="变调 (半音)",
                    info="正值升调，负值降调"
                )
                speaker_id = gr.Number(
                    value=1, label="说话人ID",
                    info="多说话人模型使用"
                )
                infer_step = gr.Slider(
                    minimum=10, maximum=100, value=50, step=5,
                    label="推理步数",
                    info="步数越多质量越好但速度越慢"
                )
        
        with gr.Row():
            inference_btn = gr.Button("🎵 开始转换", variant="primary")
            clear_btn = gr.Button("🗑️ 清空")
        
        inference_status = gr.Textbox(
            label="推理状态",
            value="等待开始转换",
            interactive=False,
            lines=2
        )
        
        output_audio = gr.Audio(
            label="输出音频",
            type="filepath"
        )
        
        # 模型选择相关函数
        def refresh_models():
            timestamps = self.get_model_timestamps()
            if timestamps:
                first_models = self.get_model_files(timestamps[0])
                return (
                    gr.Dropdown.update(choices=timestamps, value=timestamps[0]),
                    gr.Dropdown.update(choices=first_models, value=first_models[0] if first_models else None)
                )
            return (
                gr.Dropdown.update(choices=[], value=None),
                gr.Dropdown.update(choices=[], value=None)
            )
        
        def update_model_files(timestamp):
            if not timestamp:
                return gr.Dropdown.update(choices=[], value=None)
            
            model_files = self.get_model_files(timestamp)
            return gr.Dropdown.update(choices=model_files, value=model_files[0] if model_files else None)
        
        # 绑定事件
        refresh_btn.click(
            fn=refresh_models,
            outputs=[timestamp_dropdown, model_dropdown]
        )
        
        timestamp_dropdown.select(
            fn=update_model_files,
            inputs=timestamp_dropdown,
            outputs=model_dropdown
        )
        
        inference_btn.click(
            fn=self.run_inference_realtime,
            inputs=[input_audio, model_dropdown, key_change, speaker_id, infer_step, timestamp_dropdown],
            outputs=[output_audio, inference_status],
            show_progress=True,
            api_name="inference"
        )
        
        clear_btn.click(
            fn=lambda: (None, None, "等待开始转换"),
            outputs=[input_audio, output_audio, inference_status],
            show_progress=False
        )
    
    def create_monitoring_tab(self):
        """监控面板标签页"""
        gr.Markdown("## 📊 系统监控")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 💾 存储状态")
                storage_info = gr.Textbox(
                    label="存储信息",
                    value=self.get_storage_info(),
                    interactive=False,
                    lines=5
                )
            
            with gr.Column():
                gr.Markdown("### 🖥️ 系统状态")
                system_info = gr.Textbox(
                    label="系统信息",
                    value=self.get_system_info(),
                    interactive=False,
                    lines=5
                )
        
        with gr.Row():
            refresh_monitor_btn = gr.Button("🔄 刷新监控")
            clean_logs_btn = gr.Button("🗑️ 清理日志")
        
        refresh_monitor_btn.click(
            fn=self.refresh_monitoring,
            outputs=[storage_info, system_info]
        )
        clean_logs_btn.click(
            fn=self.clean_logs,
            outputs=[storage_info]
        )
    
    # 数据预处理相关方法
    def organize_data(self, source_data_dir: str, model_type: str, train_ratio: float, multi_train_ratio: float, 
                      speaker_mapping: str, audio_format: str, target_sr: int, normalize_audio: bool):
        """组织数据目录"""
        try:
            print(f"[数据组织] 开始组织数据，源目录：{source_data_dir}")
            train_ratio = float(train_ratio)
            multi_train_ratio = float(multi_train_ratio)
            target_sr = int(target_sr)
            normalize_audio = bool(normalize_audio)
            
            if not source_data_dir or not os.path.exists(source_data_dir):
                print("[数据组织] ❌ 原始数据目录不存在")
                return "❌ 原始数据目录不存在", "", "❌ 请检查原始数据目录路径"
            
            # 清理并创建目标目录
            train_dir = "data/train/audio"
            val_dir = "data/val/audio"
            if os.path.exists(train_dir):
                shutil.rmtree(train_dir)
            if os.path.exists(val_dir):
                shutil.rmtree(val_dir)
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)

            total_files = 0
            processed_files = 0
            log_messages = []
            
            used_ratio = train_ratio if model_type == "single" else multi_train_ratio

            if model_type == "single":
                print("[数据组织] 📁 处理单人模型数据...")
                log_messages.append("📁 处理单人模型数据...")
                
                all_files = []
                for root, dirs, files in os.walk(source_data_dir):
                    for file in files:
                        if file.lower().endswith(audio_format.lower()):
                            all_files.append(os.path.join(root, file))
                
                total_files = len(all_files)
                print(f"[数据组织] 找到 {total_files} 个 {audio_format} 文件")
                log_messages.append(f"找到 {total_files} 个 {audio_format} 文件")
                
                # 随机分配训练集和验证集
                import random
                random.shuffle(all_files)
                train_count = int(len(all_files) * used_ratio)
                
                train_files = all_files[:train_count]
                val_files = all_files[train_count:]
                
                # 复制文件到目标目录
                for i, file_path in enumerate(train_files):
                    self.copy_audio_file(file_path, train_dir, f"train_{i+1:04d}.wav", target_sr, normalize_audio)
                    processed_files += 1
                
                for i, file_path in enumerate(val_files):
                    self.copy_audio_file(file_path, val_dir, f"val_{i+1:04d}.wav", target_sr, normalize_audio)
                    processed_files += 1
                
                print(f"[数据组织] ✅ 单人模型数据组织完成: 训练集{len(train_files)}个，验证集{len(val_files)}个")
                log_messages.append(f"✅ 单人模型数据组织完成: 训练集{len(train_files)}个，验证集{len(val_files)}个")

            elif model_type == "multi":
                print("[数据组织] 📁 处理多人模型数据...")
                log_messages.append("📁 处理多人模型数据...")
                
                # 解析说话人映射
                import json
                try:
                    if speaker_mapping.strip():
                        spk_mapping = json.loads(speaker_mapping)
                    else:
                        # 如果没有提供映射，自动扫描数字文件夹
                        spk_mapping = {}
                        for item in os.listdir(source_data_dir):
                            item_path = os.path.join(source_data_dir, item)
                            if os.path.isdir(item_path) and item.isdigit():
                                spk_mapping[item] = int(item)
                except json.JSONDecodeError:
                    print("[数据组织] ❌ 说话人映射JSON格式错误")
                    return "❌ 说话人映射JSON格式错误", ""
                
                if not spk_mapping:
                    print("[数据组织] ❌ 未找到有效的说话人映射")
                    return "❌ 未找到有效的说话人映射", ""
                
                print(f"[数据组织] 找到说话人: {list(spk_mapping.keys())}")
                log_messages.append(f"找到说话人: {list(spk_mapping.keys())}")
                
                # 为每个说话人创建目录
                for spk_name, spk_id in spk_mapping.items():
                    spk_train_dir = os.path.join(train_dir, str(spk_id))
                    spk_val_dir = os.path.join(val_dir, str(spk_id))
                    os.makedirs(spk_train_dir, exist_ok=True)
                    os.makedirs(spk_val_dir, exist_ok=True)
                
                # 处理每个说话人的数据
                speaker_stats = {}
                for spk_name, spk_id in spk_mapping.items():
                    spk_source_dir = os.path.join(source_data_dir, spk_name)
                    if not os.path.exists(spk_source_dir):
                        print(f"[数据组织] ⚠️ 说话人目录不存在: {spk_source_dir}")
                        continue
                    
                    # 收集该说话人的音频文件
                    spk_files = []
                    for root, dirs, files in os.walk(spk_source_dir):
                        for file in files:
                            if file.lower().endswith(audio_format.lower()):
                                spk_files.append(os.path.join(root, file))
                    
                    if not spk_files:
                        print(f"[数据组织] ⚠️ 说话人 {spk_name} 没有找到音频文件")
                        continue
                    
                    # 随机分配训练集和验证集
                    import random
                    random.shuffle(spk_files)
                    spk_train_count = int(len(spk_files) * multi_train_ratio)
                    
                    spk_train_files = spk_files[:spk_train_count]
                    spk_val_files = spk_files[spk_train_count:]
                    
                    # 复制文件到目标目录
                    spk_train_dir = os.path.join(train_dir, str(spk_id))
                    spk_val_dir = os.path.join(val_dir, str(spk_id))
                    
                    for i, file_path in enumerate(spk_train_files):
                        self.copy_audio_file(file_path, spk_train_dir, f"train_{i+1:04d}.wav", target_sr, normalize_audio)
                        processed_files += 1
                    
                    for i, file_path in enumerate(spk_val_files):
                        self.copy_audio_file(file_path, spk_val_dir, f"val_{i+1:04d}.wav", target_sr, normalize_audio)
                        processed_files += 1
                    
                    total_files += len(spk_files)
                    speaker_stats[spk_name] = {
                        'total': len(spk_files),
                        'train': len(spk_train_files),
                        'val': len(spk_val_files)
                    }
                    
                    print(f"[数据组织] 说话人 {spk_name}(ID:{spk_id}): 训练集{len(spk_train_files)}个，验证集{len(spk_val_files)}个")
                    log_messages.append(f"说话人 {spk_name}(ID:{spk_id}): 训练集{len(spk_train_files)}个，验证集{len(spk_val_files)}个")
                
                print(f"[数据组织] ✅ 多人模型数据组织完成")
                log_messages.append(f"✅ 多人模型数据组织完成")

            # 生成总结
            if model_type == "single":
                summary = f"📊 数据组织完成！\n"
                summary += f"总文件数: {total_files}\n"
                summary += f"处理成功: {processed_files}\n"
                summary += f"训练集比例: {used_ratio*100:.0f}%"
            else:  # multi
                summary = f"📊 多人模型数据组织完成！\n"
                summary += f"说话人数量: {len(spk_mapping)}\n"
                summary += f"总文件数: {total_files}\n"
                summary += f"处理成功: {processed_files}\n"
                summary += f"训练集比例: {multi_train_ratio*100:.0f}%\n\n"
                summary += "各说话人统计:\n"
                for spk_name, stats in speaker_stats.items():
                    summary += f"- {spk_name}: {stats['total']}个文件 (训练:{stats['train']}, 验证:{stats['val']})\n"
            
            log_content = "\n".join(log_messages)
            print(f"[数据组织] 数据组织完成，总文件数: {total_files}，处理成功: {processed_files}")
            return "✅ 数据组织完成", summary
            
        except Exception as e:
            print(f"[数据组织] ❌ 数据组织失败: {str(e)}")
            return f"❌ 数据组织失败: {str(e)}", ""

    def copy_audio_file(self, file_path: str, target_dir: str, new_name: str, target_sr: int, normalize: bool):
        """复制音频文件到目标目录"""
        try:
            import librosa
            import soundfile as sf
            import numpy as np
            
            target_sr = int(target_sr)
            
            # 加载音频
            y, sr = librosa.load(file_path, sr=None)
            sr = int(sr)
            
            # 重采样
            if sr != target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            
            # 音频标准化
            if normalize:
                max_val = np.max(np.abs(y))
                if max_val > 0:
                    y = y / max_val
            
            # 保存文件
            output_path = os.path.join(target_dir, new_name)
            sf.write(output_path, y, target_sr)
            
            return output_path
        except ImportError:
            raise Exception("缺少音频处理库，请安装: pip install librosa soundfile")
        except Exception as e:
            raise Exception(f"复制音频文件失败: {str(e)}")

    def start_preprocessing_with_progress(self):
        """带进度条的预处理方法"""
        print("[预处理] 🚀 预处理开始")
        yield "🚀 预处理开始", 5
        
        if self.current_process and self.current_process.poll() is None:
            print("[预处理] ❌ 已有进程在运行")
            yield "❌ 已有进程在运行", 0
            return
        
        try:
            cmd = f"python preprocess.py -c {self.opt_cfg_pth}"
            print(f"[预处理] 启动命令: {cmd}")
            
            self.current_process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, bufsize=1
            )
            
            output_lines = []
            progress = 15
            
            print("[预处理] ⚙️ 预处理运行中")
            yield "⚙️ 预处理运行中", progress
            
            # 实时读取输出
            while True:
                if self.current_process.poll() is not None:
                    break
                
                try:
                    line = self.current_process.stdout.readline()
                    if line:
                        line = line.strip()
                        output_lines.append(line + "\n")
                        print(f"[预处理] {line}")
                        
                        # 解析进度
                        import re
                        progress_match = re.search(r'(\d+)%\|', line)
                        if progress_match:
                            actual_progress = int(progress_match.group(1))
                            progress = actual_progress
                            print(f"[预处理] 进度: {actual_progress}%")
                            yield f"处理中 ({actual_progress}%)", progress
                except Exception as e:
                    print(f"[预处理] 读取输出异常: {str(e)}")
                
                time.sleep(0.2)
            
            # 等待进程完全结束
            return_code = self.current_process.wait()
            
            if return_code == 0:
                print("[预处理] ✅ 预处理成功完成!")
                yield "✅ 预处理完成", 100
            else:
                print(f"[预处理] ❌ 预处理失败 (退出代码: {return_code})")
                yield "❌ 预处理失败", progress
        
        except Exception as e:
            error_msg = f"❌ 预处理启动失败: {str(e)}"
            print(f"[预处理] {error_msg}")
            yield "❌ 启动失败", 0

    def stop_preprocessing(self):
        """停止预处理并重置进度"""
        if self.current_process and self.current_process.poll() is None:
            self.current_process.terminate()
            self.current_process = None
            return "⏹️ 预处理已停止", 0
        return "ℹ️ 没有运行中的预处理", 0

    # 训练相关方法
    def update_training_config(self, batch_size: int, learning_rate: float, total_epochs: int, 
                            interval_val: int, interval_force_save: int, interval_log: int) -> None:
        """更新训练配置"""
        import yaml
        import datetime
        
        config_path = "configs/reflow.yaml"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"读取配置文件失败: {e}")
            return
            
        # 生成时间戳目录名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = f"exp/reflow-test/{timestamp}"
        
        # 更新训练参数
        config['train']['batch_size'] = batch_size
        config['train']['lr'] = learning_rate
        config['train']['epochs'] = total_epochs
        config['train']['interval_val'] = interval_val
        config['train']['interval_force_save'] = interval_force_save
        config['train']['interval_log'] = interval_log
        config['env']['expdir'] = exp_dir
        
        # 创建输出目录
        import os
        os.makedirs(exp_dir, exist_ok=True)
            
        # 保存配置
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True)
        except Exception as e:
            print(f"保存配置文件失败: {e}")

    def start_training(self, batch_size: int, learning_rate: float, total_epochs: int,
                      interval_val: int, interval_force_save: int, interval_log: int):
        """开始训练"""
        try:
            # 更新配置
            self.update_training_config(batch_size, learning_rate, total_epochs, interval_val, interval_force_save, interval_log)
            
            # 启动训练
            command = ["python", "-m", "train_reflow", "-c", "configs/reflow.yaml"]
            
            self.current_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # 启动日志监控线程
            self.training_status = "运行中"
            self.current_epoch = 0
            self.current_step = 0
            self.loss_history = []
            
            thread = threading.Thread(target=self._monitor_training_process)
            thread.daemon = True
            thread.start()
            
            return "🚀 训练已启动", "0", "0"
            
        except Exception as e:
            error_msg = f"❌ 启动训练失败: {str(e)}"
            return error_msg, "0", "0"

    def _monitor_training_process(self):
        """后台监控训练进程，解析loss和step信息"""
        if not self.current_process:
            return
        
        try:
            while self.current_process and self.current_process.poll() is None:
                try:
                    line = self.current_process.stdout.readline()
                    if line:
                        line = line.strip()
                        print(f"[训练] {line}")
                        
                        # 解析训练信息
                        self._parse_training_line(line)
                        
                        # 将日志添加到缓存
                        self.training_log_cache += line + "\n"
                        
                        # 限制日志缓存长度
                        lines = self.training_log_cache.split('\n')
                        if len(lines) > 500:
                            self.training_log_cache = '\n'.join(lines[-400:])
                            
                except:
                    pass
                    
                time.sleep(0.1)
            
            # 训练结束处理
            if self.current_process:
                return_code = self.current_process.wait()
                if return_code == 0:
                    self.training_status = "完成"
                else:
                    self.training_status = "失败"
                    
            self.current_process = None
        
        except Exception as e:
            self.training_status = "错误"

    def _parse_training_line(self, line: str):
        """解析训练日志行，提取epoch、step、loss信息"""
        try:
            import re
            line_lower = line.lower()
            
            # 解析epoch信息
            epoch_match = re.search(r'epoch[:\s]*(\d+)', line_lower)
            if epoch_match:
                self.current_epoch = int(epoch_match.group(1))
            
            # 解析step信息  
            step_match = re.search(r'step[:\s]*(\d+)', line_lower)
            if step_match:
                self.current_step = int(step_match.group(1))
            
            # 解析loss信息
            loss_patterns = [
                r'loss[:\s]*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
                r'train.*loss[:\s]*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
                r'total.*loss[:\s]*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
            ]
            
            for pattern in loss_patterns:
                loss_match = re.search(pattern, line_lower)
                if loss_match:
                    loss_value = float(loss_match.group(1))
                    # 记录loss历史
                    self.loss_history.append((self.current_step, loss_value))
                    
                    # 限制历史记录长度
                    if len(self.loss_history) > 1000:
                        self.loss_history = self.loss_history[-800:]
                    break
                    
        except:
            pass

    def stop_training(self) -> Tuple[str, str, str]:
        """停止训练"""
        process = self.current_process
        
        if process and hasattr(process, 'poll'):
            try:
                if process.poll() is None:
                    process.terminate()
                    time.sleep(2)
                    
                    if process.poll() is None:
                        process.kill()
                        time.sleep(1)
                    
                    self.current_process = None
                    self.training_status = "已停止"
                    
                    return "⏹️ 训练已停止", str(self.current_epoch), str(self.current_step)
                else:
                    self.current_process = None
                    self.training_status = "已结束"
                    return "ℹ️ 训练进程已结束", str(self.current_epoch), str(self.current_step)
                    
            except Exception as e:
                self.current_process = None
                self.training_status = "停止出错"
                return f"❌ 停止训练时出错: {str(e)}", str(self.current_epoch), str(self.current_step)
        
        return "ℹ️ 没有运行中的训练", str(self.current_epoch), str(self.current_step)

    def force_stop_training(self) -> Tuple[str, str, str]:
        """强制停止所有训练相关进程"""
        try:
            import os
            import psutil
            
            stopped_processes = []
            
            # 停止当前记录的进程
            process = self.current_process
            if process and hasattr(process, 'poll'):
                try:
                    if process.poll() is None:
                        process.kill()
                        stopped_processes.append(f"主进程 {process.pid}")
                    self.current_process = None
                except:
                    self.current_process = None
            
            # 查找并停止所有训练相关进程
            try:
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                        
                        if any(keyword in cmdline.lower() for keyword in [
                            'train_reflow.py', 'python train', 'ddsp', 'torch'
                        ]) and 'train' in cmdline.lower():
                            
                            proc.kill()
                            stopped_processes.append(f"{proc.info['name']} (PID: {proc.info['pid']})")
                            
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                        
            except:
                pass
            
            self.training_status = "已强制停止"
            
            if stopped_processes:
                result_msg = f"🆘 强制停止完成 - 已终止 {len(stopped_processes)} 个进程"
            else:
                result_msg = "🆘 强制停止完成 (未发现运行中的训练进程)"
            
            return result_msg, str(self.current_epoch), str(self.current_step)
            
        except Exception as e:
            return f"🆘 强制停止时出错: {str(e)}", str(self.current_epoch), str(self.current_step)

    def create_loss_plot(self):
        """创建loss曲线图"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            if not hasattr(self, 'loss_history') or not self.loss_history:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, 'No Loss Data\nPlease start training and refresh', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title('Training Loss Curve')
                ax.set_xlabel('Step')
                ax.set_ylabel('Loss')
                plt.tight_layout()
                return fig
            
            # 提取step和loss数据
            steps = [x[0] for x in self.loss_history]
            losses = [x[1] for x in self.loss_history]
            
            # 创建图表
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(steps, losses, 'b-', linewidth=1.5, alpha=0.8)
            ax.set_title(f'Training Loss Curve (Epoch: {self.current_epoch}, Step: {self.current_step})')
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
            
            if losses:
                latest_loss = losses[-1]
                ax.text(0.02, 0.98, f'Latest Loss: {latest_loss:.6f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'Plot Error:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12, color='red')
            ax.set_title('Training Loss Curve (Error)')
            return fig

    def get_training_info(self):
        """获取训练信息"""
        if hasattr(self, 'current_process') and self.current_process:
            if self.current_process.poll() is None:
                status = f"🏃 训练运行中 (PID: {self.current_process.pid})"
            else:
                status = "🔚 训练进程已结束"
        elif hasattr(self, 'training_status'):
            if self.training_status == "完成":
                status = "✅ 训练已完成"
            elif self.training_status == "失败":
                status = "❌ 训练失败"
            elif self.training_status == "已停止":
                status = "⏹️ 训练已停止"
            else:
                status = "💤 未开始训练"
        else:
            status = "💤 未开始训练"
        
        epoch = str(self.current_epoch) if hasattr(self, 'current_epoch') else "0"
        step = str(self.current_step) if hasattr(self, 'current_step') else "0"
        loss_fig = self.create_loss_plot()
        
        return status, epoch, step, loss_fig

    # 推理相关方法
    def get_model_timestamps(self) -> list:
        """获取所有训练时间戳目录"""
        base_dir = "exp/reflow-test"
        if not os.path.exists(base_dir):
            return []
            
        timestamp_dirs = sorted(glob.glob(os.path.join(base_dir, "*")), reverse=True)
        timestamps = []
        for dir_path in timestamp_dirs:
            if os.path.isdir(dir_path):
                timestamp = os.path.basename(dir_path)
                timestamps.append(timestamp)
                
        return timestamps
        
    def get_model_files(self, timestamp: str) -> list:
        """获取指定时间戳目录下的所有模型文件"""
        if not timestamp:
            return []
            
        model_dir = os.path.join("exp/reflow-test", timestamp)
        if not os.path.exists(model_dir):
            return []
            
        model_files = glob.glob(os.path.join(model_dir, "*.pt"))
        
        if not model_files:
            return []
        
        model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        model_files = [os.path.basename(f) for f in model_files]
        
        return model_files

    def run_inference_realtime(self, input_audio, model_file, key_change, speaker_id, infer_step, timestamp):
        """实时推理，控制台实时输出，界面显示状态"""
        try:
            if not timestamp or not model_file:
                error_msg = "❌ 请先选择模型文件"
                print(f"[推理] {error_msg}")
                return None, error_msg
            
            if not input_audio:
                error_msg = "❌ 请先上传音频文件"
                print(f"[推理] {error_msg}")
                return None, error_msg
            
            # 构建完整的模型路径
            model_path = os.path.join("exp/reflow-test", timestamp, model_file)
            print(f"[推理] 使用模型: {model_path}")
            
            if not os.path.exists(model_path):
                error_msg = f"❌ 模型文件不存在: {model_path}"
                print(f"[推理] {error_msg}")
                return None, error_msg
            
            # 生成输出文件名
            output_path = f"results/converted_{int(time.time())}_{os.path.basename(input_audio)}"
            
            # 启动推理
            command = ["python", "-m", "main_reflow",
                      "-i", input_audio,
                      "-m", model_path,
                      "-o", output_path,
                      "-k", str(int(key_change)),
                      "-id", str(int(speaker_id)),
                      "-step", str(int(infer_step))]
            
            print(f"[推理] 启动命令: {' '.join(command)}")
            print(f"[推理] 🚀 开始推理转换...")
            
            # 使用Popen实时读取输出
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            current_status = "🚀 推理已启动，正在初始化..."
            
            # 实时读取输出
            while True:
                try:
                    line = process.stdout.readline()
                    if line:
                        line = line.strip()
                        print(f"[推理] {line}")
                        
                        # 更新状态显示
                        if "sample time step:" in line and "%" in line:
                            import re
                            match = re.search(r'sample time step:\s*(\d+)%', line)
                            if match:
                                sample_progress = int(match.group(1))
                                current_status = f"🔄 采样中: {sample_progress}%"
                        elif "loading" in line.lower() or "load" in line.lower():
                            current_status = "📂 加载模型中..."
                        elif "processing" in line.lower() or "process" in line.lower():
                            current_status = "⚙️ 处理音频中..."
                        elif "saving" in line.lower() or "save" in line.lower():
                            current_status = "💾 保存结果中..."
                        elif "error" in line.lower() or "failed" in line.lower():
                            current_status = f"❌ 处理出错: {line[:50]}..."
                    else:
                        if process.poll() is not None:
                            break
                except:
                    if process.poll() is not None:
                        break
                
                time.sleep(0.1)
            
            # 等待进程完全结束
            return_code = process.wait()
            
            # 检查结果
            if return_code == 0 and os.path.exists(output_path):
                success_msg = "✅ 转换完成！"
                print(f"[推理] {success_msg} 输出文件: {output_path}")
                return output_path, success_msg
            else:
                if return_code != 0:
                    error_msg = f"❌ 转换失败 (退出代码: {return_code})"
                else:
                    error_msg = "❌ 转换失败，输出文件未生成"
                print(f"[推理] {error_msg}")
                return None, error_msg
                
        except Exception as e:
            error_msg = f"❌ 转换出错: {str(e)}"
            print(f"[推理] {error_msg}")
            return None, error_msg

    # 监控相关方法
    def get_storage_info(self) -> str:
        """获取存储信息"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(".")
            
            info = f"磁盘空间:\n"
            info += f"总计: {total // (1024**3):.1f} GB\n"
            info += f"已用: {used // (1024**3):.1f} GB\n"
            info += f"可用: {free // (1024**3):.1f} GB\n\n"
            
            # 检查关键目录
            dirs = ['data/train/audio', 'data/val/audio', 'exp', 'results']
            for dir_path in dirs:
                if os.path.exists(dir_path):
                    files = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
                    info += f"{dir_path}: {files} 个文件\n"
                else:
                    info += f"{dir_path}: 不存在\n"
            
            return info
        except Exception as e:
            return f"获取存储信息失败: {str(e)}"
    
    def get_system_info(self) -> str:
        """获取系统信息"""
        try:
            import torch
            import psutil
            
            info = f"Python: {sys.version.split()[0]}\n"
            info += f"PyTorch: {torch.__version__}\n"
            
            if torch.cuda.is_available():
                info += f"CUDA: {torch.version.cuda}\n"
                info += f"GPU: {torch.cuda.get_device_name(0)}\n"
                info += f"显存: {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB\n"
            else:
                info += "CUDA: 不可用\n"
            
            info += f"CPU: {psutil.cpu_count()} 核心\n"
            info += f"内存: {psutil.virtual_memory().total // (1024**3)} GB\n"
            
            return info
        except Exception as e:
            return f"获取系统信息失败: {str(e)}"
    
    def refresh_monitoring(self) -> Tuple[str, str]:
        """刷新监控信息"""
        return self.get_storage_info(), self.get_system_info()
    
    def clean_logs(self) -> str:
        """清理日志"""
        self.log_content = ""
        return "✅ 日志已清理"
    
    def create_msst_tab(self):
        """MSST 音频分离标签页"""
        gr.Markdown("## 🎼 音频分离 (MSST)")
        gr.Markdown("使用 MSST 模型进行音频源分离，支持人声、乐器等多种分离模式")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📁 输入设置")
                msst_input_type = gr.Radio(
                    choices=["单个文件", "文件夹批处理"],
                    value="单个文件",
                    label="输入类型"
                )
                
                # 单个文件输入
                with gr.Group(visible=True) as single_file_group:
                    msst_input_audio = gr.Audio(
                        label="输入音频文件",
                        type="filepath"
                    )
                
                # 文件夹批处理输入
                with gr.Group(visible=False) as folder_input_group:
                    msst_input_folder = gr.Textbox(
                        label="输入文件夹路径",
                        placeholder="例如: input/audio_files",
                        value="input"
                    )
                
                msst_output_folder = gr.Textbox(
                    label="输出文件夹",
                    placeholder="例如: results/msst",
                    value="results/msst"
                )
            
            with gr.Column():
                gr.Markdown("### ⚙️ 模型设置")
                msst_model_type = gr.Dropdown(
                    choices=self.get_msst_model_types(),
                    value=self.get_msst_model_types()[0] if self.get_msst_model_types() else "bs_roformer",
                    label="模型类型",
                    info="选择要使用的分离模型类型"
                )
                
                msst_model_name = gr.Dropdown(
                    choices=[],
                    label="模型名称",
                    info="选择具体的模型",
                    interactive=True
                )
                
                with gr.Row():
                    msst_refresh_models_btn = gr.Button("🔄 刷新模型列表", size="sm")
                
                # 隐藏路径显示框，但保留变量供内部使用
                msst_model_path = gr.Textbox(
                    label="模型文件路径",
                    placeholder="将根据模型选择自动填充",
                    info="模型检查点文件的完整路径",
                    interactive=False,
                    visible=False  # 隐藏显示
                )
                
                msst_config_path = gr.Textbox(
                    label="配置文件路径",
                    placeholder="将根据模型选择自动填充",
                    info="模型配置文件路径",
                    interactive=False,
                    visible=False  # 隐藏显示
                )
                
                # 新增：分离目标选择
                with gr.Group() as config_group:
                    gr.Markdown("### 🎯 分离目标设置")
                    msst_target_instrument = gr.Radio(
                        choices=[],
                        label="目标乐器",
                        info="选择要分离的目标乐器类型",
                        visible=False
                    )
                    
                    msst_config_status = gr.Textbox(
                        label="配置状态",
                        value="请先选择模型",
                        interactive=False,
                        lines=2
                    )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔧 分离参数")
                msst_device = gr.Dropdown(
                    choices=["auto", "cpu", "cuda", "mps"],
                    value="auto",
                    label="计算设备",
                    info="选择用于推理的设备"
                )
                
                msst_output_format = gr.Dropdown(
                    choices=["wav", "flac", "mp3"],
                    value="wav",
                    label="输出格式"
                )
                
                msst_use_tta = gr.Checkbox(
                    value=False,
                    label="使用 TTA (测试时增强)",
                    info="提高质量但增加处理时间"
                )
            
            with gr.Column():
                gr.Markdown("### 📊 音频参数")
                msst_wav_bit_depth = gr.Dropdown(
                    choices=["PCM_16", "PCM_24", "PCM_32", "FLOAT"],
                    value="FLOAT",
                    label="WAV 位深度"
                )
                
                msst_flac_bit_depth = gr.Dropdown(
                    choices=["PCM_16", "PCM_24"],
                    value="PCM_24",
                    label="FLAC 位深度"
                )
                
                msst_mp3_bit_rate = gr.Dropdown(
                    choices=["96k", "128k", "192k", "256k", "320k"],
                    value="320k",
                    label="MP3 比特率"
                )
        
        with gr.Row():
            msst_start_btn = gr.Button("🎵 开始分离", variant="primary")
            msst_stop_btn = gr.Button("⏹️ 停止分离", variant="stop")
            msst_refresh_status_btn = gr.Button("🔄 刷新状态", variant="secondary")
        
        msst_status = gr.Textbox(
            label="分离状态",
            value="等待开始分离",
            interactive=False,
            lines=3
        )
        
        # 输出音频预览（仅单文件模式）
        with gr.Group(visible=True) as output_audio_group:
            gr.Markdown("### 🎵 分离结果预览")
            msst_output_audio = gr.Audio(
                label="分离后的音频",
                type="filepath"
            )
        
        # 输入类型切换事件
        def toggle_input_type(input_type):
            if input_type == "单个文件":
                return (
                    gr.Group(visible=True),
                    gr.Group(visible=False),
                    gr.Group(visible=True)
                )
            else:
                return (
                    gr.Group(visible=False),
                    gr.Group(visible=True),
                    gr.Group(visible=False)
                )
        
        msst_input_type.change(
            fn=toggle_input_type,
            inputs=[msst_input_type],
            outputs=[single_file_group, folder_input_group, output_audio_group]
        )
        
        # 模型类型变化时更新模型列表
        def update_model_list(model_type):
            models = self.get_msst_models_for_type(model_type)
            if models:
                return gr.Dropdown(choices=models, value=models[0])
            else:
                return gr.Dropdown(choices=[], value=None)
        
        # 模型选择变化时更新文件路径和配置信息
        def update_model_paths_and_config(model_type, model_name):
            if model_type and model_name:
                model_path, config_path = self.get_model_files_path(model_type, model_name)
                
                # 解析配置文件
                instruments, target_instrument = self.parse_config_file(config_path)
                
                if instruments:
                    return (
                        model_path, 
                        config_path,
                        gr.Radio(choices=instruments, value=target_instrument, visible=True),
                        f"✅ 模型加载成功\n📁 模型: {model_name}\n🎯 可用乐器: {', '.join(instruments)}\n🎵 当前目标: {target_instrument}"
                    )
                else:
                    return (
                        model_path, 
                        config_path,
                        gr.Radio(choices=[], visible=False),
                        "配置文件解析失败或无training配置"
                    )
            return "", "", gr.Radio(choices=[], visible=False), "请先选择模型"
        
        # 目标乐器变化时更新配置文件
        def update_target_instrument(config_path, new_target):
            # 添加更严格的检查，避免在刷新或初始化时显示错误
            if not config_path or not new_target:
                return ""  # 返回空字符串，不显示任何信息
            
            # 检查配置文件是否存在
            if not os.path.exists(config_path):
                return ""  # 配置文件不存在时也不显示错误
            
            # 只有在配置路径和目标乐器都有效时才尝试更新
            success, message = self.update_config_target_instrument(config_path, new_target)
            if success:
                return f"✅ {message}"
            else:
                return f"❌ {message}"
        
        # 刷新模型列表
        def refresh_models(model_type):
            models = self.get_msst_models_for_type(model_type)
            if models:
                return gr.Dropdown(choices=models, value=models[0])
            else:
                return gr.Dropdown(choices=[], value=None)
        
        # 刷新模型列表（修复版本）
        def refresh_models_only(model_type):
            """只刷新模型列表，不触发配置更新"""
            models = self.get_msst_models_for_type(model_type)
            if models:
                return gr.Dropdown(choices=models, value=models[0])
            else:
                return gr.Dropdown(choices=[], value=None)
        
        # 绑定事件
        msst_model_type.change(
            fn=update_model_list,
            inputs=[msst_model_type],
            outputs=[msst_model_name]
        ).then(
            fn=update_model_paths_and_config,
            inputs=[msst_model_type, msst_model_name],
            outputs=[msst_model_path, msst_config_path, msst_target_instrument, msst_config_status]
        )
        msst_refresh_status_btn.click(
            fn=self.get_msst_status,
            outputs=[msst_status]
        )
        
        msst_model_name.change(
            fn=update_model_paths_and_config,
            inputs=[msst_model_type, msst_model_name],
            outputs=[msst_model_path, msst_config_path, msst_target_instrument, msst_config_status]
        )
        
        msst_target_instrument.change(
            fn=update_target_instrument,
            inputs=[msst_config_path, msst_target_instrument],
            outputs=[msst_config_status]
        )
        
        # 修复：刷新按钮只刷新模型列表，不触发配置更新
        msst_refresh_models_btn.click(
            fn=refresh_models_only,
            inputs=[msst_model_type],
            outputs=[msst_model_name]
        )
        
        # 绑定分离事件 - 使用生成器实现实时更新
        msst_start_btn.click(
            fn=self.start_msst_separation_with_updates,
            inputs=[
                msst_input_type, msst_input_audio, msst_input_folder,
                msst_output_folder, msst_model_type, msst_model_path,
                msst_config_path, msst_device, msst_output_format,
                msst_use_tta, msst_wav_bit_depth, msst_flac_bit_depth,
                msst_mp3_bit_rate, msst_target_instrument
            ],
            outputs=[msst_status, msst_output_audio],
            show_progress=True
        )
        
        msst_stop_btn.click(
            fn=self.stop_msst_separation,
            outputs=[msst_status]
        )
        
        # msst_refresh_btn.click(
        #     fn=self.get_msst_status,
        #     outputs=[msst_status]
        # )
    
    def parse_config_file(self, config_path):
        """解析配置文件并提取training信息"""
        if not config_path or not os.path.exists(config_path):
            return [], None
        
        try:
            # 自定义YAML加载器来处理Python特定标签
            class SafeLoaderIgnorePython(yaml.SafeLoader):
                pass
            
            def python_tuple_constructor(loader, node):
                """处理 !!python/tuple 标签"""
                return tuple(loader.construct_sequence(node))
            
            def python_object_constructor(loader, node):
                """处理其他Python对象标签，返回原始值"""
                if isinstance(node, yaml.SequenceNode):
                    return loader.construct_sequence(node)
                elif isinstance(node, yaml.MappingNode):
                    return loader.construct_mapping(node)
                else:
                    return loader.construct_scalar(node)
            
            # 注册自定义构造器
            SafeLoaderIgnorePython.add_constructor(
                'tag:yaml.org,2002:python/tuple', 
                python_tuple_constructor
            )
            SafeLoaderIgnorePython.add_constructor(
                'tag:yaml.org,2002:python/object', 
                python_object_constructor
            )
            SafeLoaderIgnorePython.add_constructor(
                'tag:yaml.org,2002:python/object/apply', 
                python_object_constructor
            )
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.load(f, Loader=SafeLoaderIgnorePython)
            
            self.current_config_data = config_data
            
            if 'training' in config_data:
                training_config = config_data['training']
                instruments = training_config.get('instruments', [])
                target_instrument = training_config.get('target_instrument', '')
                return instruments, target_instrument
            
            return [], None
            
        except Exception as e:
            print(f"解析配置文件失败: {e}")
            # 尝试备用方案：直接读取文件内容并查找training部分
            try:
                return self._parse_config_fallback(config_path)
            except Exception as fallback_error:
                print(f"备用解析方案也失败: {fallback_error}")
                return [], None
    
    def _parse_config_fallback(self, config_path):
        """备用配置解析方案：使用正则表达式提取training信息"""
        import re
        
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找training部分
        training_match = re.search(r'training:\s*\n((?:  .+\n)*)', content)
        if not training_match:
            return [], None
        
        training_section = training_match.group(1)
        
        # 提取instruments列表
        instruments = []
        instruments_match = re.search(r'instruments:\s*\n((?:  - .+\n)*)', training_section)
        if instruments_match:
            instruments_lines = instruments_match.group(1)
            instruments = re.findall(r'- (\w+)', instruments_lines)
        
        # 提取target_instrument
        target_match = re.search(r'target_instrument:\s*(\w+)', training_section)
        target_instrument = target_match.group(1) if target_match else ''
        
        return instruments, target_instrument
    
    def update_config_target_instrument(self, config_path, new_target):
        """更新配置文件中的target_instrument"""
        if not config_path or not os.path.exists(config_path):
            return False, "配置文件不存在"
        
        try:
            # 读取原始文件内容
            with open(config_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 逐行查找并替换target_instrument
            modified = False
            for i, line in enumerate(lines):
                if 'target_instrument:' in line and not line.strip().startswith('#'):
                    # 保持原有的缩进格式
                    indent = len(line) - len(line.lstrip())
                    lines[i] = ' ' * indent + f'target_instrument: {new_target}\n'
                    modified = True
                    break
            
            if modified:
                # 写回文件
                with open(config_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                return True, f"已将target_instrument更新为: {new_target}"
            else:
                return False, "未找到target_instrument字段"
            
        except Exception as e:
            return False, f"更新配置文件失败: {e}"


    def stop_msst_separation(self):
        """停止 MSST 音频分离"""
        try:
            if hasattr(self, 'msst_separator'):
                self.msst_separator.del_cache()
                delattr(self, 'msst_separator')
            
            self.msst_status = "⏹️ 分离已停止"
            return "⏹️ 分离已停止"
        except Exception as e:
            return f"❌ 停止分离失败: {str(e)}"
    
    def get_msst_model_types(self):
        """获取可用的MSST模型类型"""
        model_types = []
        if os.path.exists(self.msst_pretrain_dir):
            for item in os.listdir(self.msst_pretrain_dir):
                item_path = os.path.join(self.msst_pretrain_dir, item)
                if os.path.isdir(item_path):
                    model_types.append(item)
        return model_types if model_types else ["bs_roformer", "mel_band_roformer", "htdemucs", "mdx23c", "segm_models"]
    
    def get_msst_models_for_type(self, model_type):
        """获取指定模型类型下的可用模型"""
        models = []
        type_dir = os.path.join(self.msst_pretrain_dir, model_type)
        if os.path.exists(type_dir):
            for item in os.listdir(type_dir):
                item_path = os.path.join(type_dir, item)
                if os.path.isdir(item_path):
                    # 检查是否包含.ckpt文件
                    ckpt_files = [f for f in os.listdir(item_path) if f.endswith('.ckpt')]
                    if ckpt_files:
                        models.append(item)
        return models
    
    def get_model_files_path(self, model_type, model_name):
        """获取模型文件和配置文件的完整路径"""
        model_dir = os.path.join(self.msst_pretrain_dir, model_type, model_name)
        model_path = ""
        config_path = ""
        
        if os.path.exists(model_dir):
            # 查找.ckpt文件
            ckpt_files = [f for f in os.listdir(model_dir) if f.endswith('.ckpt')]
            if ckpt_files:
                model_path = os.path.join(model_dir, ckpt_files[0])
            
            # 查找.yaml文件
            yaml_files = [f for f in os.listdir(model_dir) if f.endswith('.yaml')]
            if yaml_files:
                config_path = os.path.join(model_dir, yaml_files[0])
        
        return model_path, config_path
    
    def get_msst_status(self):
        """获取当前MSST分离状态"""
        return getattr(self, 'msst_status', '等待开始分离')
    
    def start_msst_separation_with_updates(self, input_type, input_audio, input_folder, output_folder,
                                          model_type, model_path, config_path, device, output_format,
                                          use_tta, wav_bit_depth, flac_bit_depth, mp3_bit_rate, target_instrument):
        """带实时更新的MSST音频分离"""
        try:
            # 验证输入
            if input_type == "单个文件":
                if not input_audio:
                    return "❌ 请上传音频文件", None
                input_path = input_audio
            else:
                if not input_folder or not os.path.exists(input_folder):
                    return "❌ 请提供有效的输入文件夹路径", None
                input_path = input_folder
            
            if not model_path or not os.path.exists(model_path):
                return "❌ 请提供有效的模型文件路径", None
            
            # 创建输出目录
            os.makedirs(output_folder, exist_ok=True)
            
            # 构建store_dirs
            if not config_path:
                config_path = model_path.replace("pretrain", "configs") + ".yaml"
            
            # 加载配置文件获取instruments信息
            from tools.msst.utils.merged_utils import load_configs
            model_config = load_configs(config_path)
            
            # 构建store_dirs字典
            if target_instrument and target_instrument.strip():
                # 如果选择了target_instrument，只输出该乐器
                store_dirs = {}
                if target_instrument in model_config.training.get("instruments", []):
                    store_dirs[target_instrument] = output_folder
                else:
                    yield f"❌ 选择的乐器 '{target_instrument}' 不在模型支持的乐器列表中", None
                    return
                
                if not store_dirs:
                    yield "❌ 未找到有效的目标乐器", None
                    return
            else:
                # 如果未选择target_instrument，输出所有乐器
                store_dirs = output_folder
            
            # 设置音频参数
            audio_params = {
                "wav_bit_depth": wav_bit_depth,
                "flac_bit_depth": flac_bit_depth,
                "mp3_bit_rate": mp3_bit_rate
            }
            
            # 初始化状态
            self.msst_status = "🔧 正在初始化分离器..."
            yield self.msst_status, None
            
            # 创建分离器
            logger = get_logger(console_level=logging.INFO)
            
            self.msst_separator = MSSeparator(
                model_type=model_type,
                config_path=config_path,
                model_path=model_path,
                device=device,
                output_format=output_format,
                use_tta=use_tta,
                store_dirs=store_dirs,
                audio_params=audio_params,
                logger=logger
            )
            
            self.msst_status = "📥 正在加载音频文件..."
            yield self.msst_status, None
            
            if input_type == "单个文件":
                # 单文件处理
                import librosa
                import soundfile as sf
                
                # 加载音频
                mix, sr = librosa.load(input_path, sr=44100, mono=False)
                
                self.msst_status = "🎵 正在执行音频分离..."
                yield self.msst_status, None
                
                # 执行分离
                results = self.msst_separator.separate(mix)
                
                self.msst_status = "💾 正在保存分离结果..."
                yield self.msst_status, None
                
                # 保存结果
                file_name = os.path.splitext(os.path.basename(input_path))[0]
                output_files = []
                
                # 根据store_dirs字典决定保存哪些乐器
                if isinstance(store_dirs, dict):
                    # 如果store_dirs是字典，只保存字典中指定的乐器
                    for instr in results.keys():
                        if instr in store_dirs:
                            output_dir = store_dirs[instr]
                            output_file = os.path.join(output_dir, f"{file_name}_{instr}.{output_format}")
                            self.msst_separator.save_audio(results[instr], sr, f"{file_name}_{instr}", output_dir)
                            output_files.append(output_file)
                else:
                    # 如果store_dirs是字符串，保存所有乐器到该目录
                    for instr in results.keys():
                        output_file = os.path.join(store_dirs, f"{file_name}_{instr}.{output_format}")
                        self.msst_separator.save_audio(results[instr], sr, f"{file_name}_{instr}", store_dirs)
                        output_files.append(output_file)
                
                self.msst_status = f"✅ 分离完成！输出文件: {', '.join([os.path.basename(f) for f in output_files])}"
                yield self.msst_status, output_files[0] if output_files else None
            else:
                # 文件夹批处理
                self.msst_status = "📁 正在批量处理文件夹..."
                yield self.msst_status, None
                
                success_files = self.msst_separator.process_folder(input_path)
                
                self.msst_status = f"✅ 批处理完成！成功处理 {len(success_files)} 个文件"
                yield self.msst_status, None
                
        except Exception as e:
            self.msst_status = f"❌ 分离失败: {str(e)}"
            yield self.msst_status, None
        finally:
            if hasattr(self, 'msst_separator'):
                self.msst_separator.del_cache()

def main():
    """主程序入口"""
    webui = EnhancedWebUI()
    app = webui.create_interface()
    app.queue()
    app.launch(
        server_name="0.0.0.0",
        server_port=7865,
        share=True,
        inbrowser=False
    )

if __name__ == "__main__":
    main()