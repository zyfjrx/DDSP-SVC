# coding: utf-8
__author__ = "Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/"
__license__ = "AGPL-3.0"
__author__ = "Sucial https://github.com/SUC-DriverOld"

import gc
import os
import sys
import json
import locale
import platform
import yaml
import gradio as gr
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from ml_collections import ConfigDict
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from numpy.typing import NDArray
from typing import Dict
from pathlib import Path
# ROOT_DIR = Path(__file__).parent.parent
# sys.path.append(str(ROOT_DIR))
from tools.msst.utils.logger import get_logger, set_log_level
from tools.msst.utils.constant import *
from tools.msst.tools.i18n.i18n import I18nAuto

logger = get_logger()


# ============================================================================
# 原 utils/utils.py 的内容
# ============================================================================

def get_model_from_config(model_type, config_path):
	with open(config_path) as f:
		if model_type == "htdemucs":
			config = OmegaConf.load(config_path)
		else:
			config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

	if model_type == "mdx23c":
		from modules.mdx23c_tfc_tdf_v3 import TFC_TDF_net

		model = TFC_TDF_net(config)
	elif model_type == "htdemucs":
		from modules.demucs4ht import get_model

		model = get_model(config)
	elif model_type == "segm_models":
		from modules.segm_models import Segm_Models_Net

		model = Segm_Models_Net(config)
	elif model_type == "torchseg":
		from modules.torchseg_models import Torchseg_Net

		model = Torchseg_Net(config)
	elif model_type == "mel_band_roformer":
		from modules.bs_roformer import MelBandRoformer

		model = MelBandRoformer(**dict(config.model))
	elif model_type == "bs_roformer":
		from modules.bs_roformer import BSRoformer

		model = BSRoformer(**dict(config.model))
	elif model_type == "swin_upernet":
		from modules.upernet_swin_transformers import Swin_UperNet_Model

		model = Swin_UperNet_Model(config)
	elif model_type == "bandit":
		from modules.bandit.core.model import MultiMaskMultiSourceBandSplitRNNSimple

		model = MultiMaskMultiSourceBandSplitRNNSimple(**config.model)
	elif model_type == "bandit_v2":
		from modules.bandit_v2.bandit import Bandit

		model = Bandit(**config.kwargs)
	elif model_type == "scnet_unofficial":
		from modules.scnet_unofficial import SCNet

		model = SCNet(**config.model)
	elif model_type == "scnet":
		from modules.scnet import SCNet

		model = SCNet(**config.model)
	elif model_type == "apollo":
		from modules.look2hear import BaseModel

		model = BaseModel.apollo(**config.model)
	elif model_type == "bs_mamba2":
		from modules.ts_bs_mamba2 import Separator

		model = Separator(**config.model)
	else:
		logger.error("Unknown model: {}".format(model_type))
		model = None

	return model, config


def demix(config, model, mix: NDArray, device, model_type: str = None, callback=None) -> Dict[str, NDArray]:
	mix = torch.tensor(mix, dtype=torch.float32)

	C = config.audio.chunk_size if model_type != "htdemucs" else config.training.samplerate * config.training.segment
	N = config.inference.num_overlap
	batch_size = config.inference.batch_size
	step = int(C // N)

	# HTDemucs doesn't use border padding and fading
	use_fading = model_type != "htdemucs"

	if use_fading:
		fade_size = C // 10
		border = C - step
	else:
		border = 0

	length_init = mix.shape[-1]

	# Apply padding for non-HTDemucs models
	if use_fading and length_init > 2 * border and (border > 0):
		if mix.ndim == 1:
			mix = mix.unsqueeze(0)  # [1, length]
		mix = nn.functional.pad(mix, (border, border), mode="reflect")

	# Prepare windows arrays for non-HTDemucs models
	if use_fading:
		window_size = C
		fadein = torch.linspace(0, 1, fade_size)
		fadeout = torch.linspace(1, 0, fade_size)
		window_start = torch.ones(window_size)
		window_middle = torch.ones(window_size)
		window_finish = torch.ones(window_size)
		window_start[-fade_size:] *= fadeout  # First audio chunk, no fadein
		window_finish[:fade_size] *= fadein  # Last audio chunk, no fadeout
		window_middle[-fade_size:] *= fadeout
		window_middle[:fade_size] *= fadein

	with torch.amp.autocast("cuda", enabled=config.training.get("use_amp", True)):
		with torch.inference_mode():
			# Determine the shape of the result based on model type and configuration
			if model_type == "htdemucs":
				S = len(config.training.instruments)
				req_shape = (S,) + tuple(mix.shape)
			else:
				if config.training.target_instrument is not None:
					req_shape = (1,) + tuple(mix.shape)
				else:
					req_shape = (len(config.training.instruments),) + tuple(mix.shape)

			result = torch.zeros(req_shape, dtype=torch.float32)
			counter = torch.zeros(req_shape, dtype=torch.float32)
			i = 0
			batch_data = []
			batch_locations = []
			progress_bar = tqdm(total=mix.shape[1], desc="Processing audio chunks", leave=False)

			while i < mix.shape[1]:
				part = mix[:, i : i + C].to(device)
				length = part.shape[-1]

				# Pad the last chunk if needed
				if length < C:
					if use_fading and length > C // 2 + 1:
						part = nn.functional.pad(input=part, pad=(0, C - length), mode="reflect")
					else:
						part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode="constant", value=0)

				batch_data.append(part)
				batch_locations.append((i, length))
				i += step

				if len(batch_data) >= batch_size or (i >= mix.shape[1]):
					arr = torch.stack(batch_data, dim=0)
					x = model(arr)

					for j in range(len(batch_locations)):
						start, l = batch_locations[j]

						if use_fading:
							# Apply windowing for regular model
							window = window_middle
							if i - step == 0:  # First audio chunk
								window = window_start
							elif i >= mix.shape[1]:  # Last audio chunk
								window = window_finish

							result[..., start : start + l] += x[j][..., :l].cpu() * window[..., :l]
							counter[..., start : start + l] += window[..., :l]
						else:
							# Simple accumulation for HTDemucs
							result[..., start : start + l] += x[j][..., :l].cpu()
							counter[..., start : start + l] += 1.0

					batch_data = []
					batch_locations = []

				progress_bar.update(step)

				if callback:
					callback["progress"] = min(0.99 * (i / mix.shape[1]), 0.99)  # the rest 1% is for the postprocess

			progress_bar.close()

			estimated_sources = result / counter
			estimated_sources = estimated_sources.cpu().numpy()
			np.nan_to_num(estimated_sources, copy=False, nan=0.0)

			# Remove padding for non-HTDemucs models
			if use_fading and length_init > 2 * border and (border > 0):
				estimated_sources = estimated_sources[..., border:-border]

	# Return the results based on configuration
	if model_type == "htdemucs":
		if len(config.training.instruments) > 1:
			return {k: v for k, v in zip(config.training.instruments, estimated_sources)}
		else:
			return estimated_sources
	else:  # Regular model
		if config.training.target_instrument is None:
			return {k: v for k, v in zip(config.training.instruments, estimated_sources)}
		else:
			return {k: v for k, v in zip([config.training.target_instrument], estimated_sources)}


def sdr(references, estimates):
	# compute SDR for one song
	delta = 1e-7  # avoid numerical errors
	num = np.sum(np.square(references), axis=(1, 2))
	den = np.sum(np.square(references - estimates), axis=(1, 2))
	num += delta
	den += delta
	return 10 * np.log10(num / den)


def si_sdr(reference, estimate):
	eps = 1e-07
	scale = np.sum(estimate * reference + eps, axis=(0, 1)) / np.sum(reference**2 + eps, axis=(0, 1))
	scale = np.expand_dims(scale, axis=(0, 1))  # shape - [50, 1]
	reference = reference * scale
	sisdr = np.mean(10 * np.log10(np.sum(reference**2, axis=(0, 1)) / (np.sum((reference - estimate) ** 2, axis=(0, 1)) + eps) + eps))
	return sisdr


def L1Freq_metric(reference, estimate, fft_size=2048, hop_size=1024, device="cpu"):
	reference = torch.from_numpy(reference).to(device)
	estimate = torch.from_numpy(estimate).to(device)
	reference_stft = torch.stft(reference, fft_size, hop_size, return_complex=True)
	estimated_stft = torch.stft(estimate, fft_size, hop_size, return_complex=True)
	reference_mag = torch.abs(reference_stft)
	estimate_mag = torch.abs(estimated_stft)
	loss = 10 * F.l1_loss(estimate_mag, reference_mag)
	# Metric is on the range from 0 to 100 - larger the better
	ret = 100 / (1.0 + float(loss.cpu().numpy()))
	return ret


def LogWMSE_metric(reference, estimate, mixture, device="cpu"):
	from torch_log_wmse import LogWMSE

	log_wmse = LogWMSE(
		audio_length=reference.shape[-1] / 44100,
		sample_rate=44100,
		return_as_loss=False,  # optional
		bypass_filter=False,  # optional
	)
	reference = torch.from_numpy(reference).unsqueeze(0).unsqueeze(0).to(device)
	estimate = torch.from_numpy(estimate).unsqueeze(0).unsqueeze(0).to(device)
	mixture = torch.from_numpy(mixture).unsqueeze(0).to(device)
	# logger.info(reference.shape, estimate.shape, mixture.shape)
	res = log_wmse(mixture, reference, estimate)
	return float(res.cpu().numpy())


def AuraSTFT_metric(reference, estimate, device="cpu"):
	from auraloss.freq import STFTLoss

	stft_loss = STFTLoss(w_log_mag=1.0, w_lin_mag=0.0, w_sc=1.0, device=device)
	reference = torch.from_numpy(reference).unsqueeze(0).to(device)
	estimate = torch.from_numpy(estimate).unsqueeze(0).to(device)
	res = 100 / (1.0 + 10 * stft_loss(reference, estimate))
	return float(res.cpu().numpy())


def AuraMRSTFT_metric(reference, estimate, device="cpu"):
	from auraloss.freq import MultiResolutionSTFTLoss

	mrstft_loss = MultiResolutionSTFTLoss(
		fft_sizes=[1024, 2048, 4096], hop_sizes=[256, 512, 1024], win_lengths=[1024, 2048, 4096], scale="mel", n_bins=128, sample_rate=44100, perceptual_weighting=True, device=device
	)
	reference = torch.from_numpy(reference).unsqueeze(0).float().to(device)
	estimate = torch.from_numpy(estimate).unsqueeze(0).float().to(device)
	res = 100 / (1.0 + 10 * mrstft_loss(reference, estimate))
	return float(res.cpu().numpy())


def bleed_full(reference, estimate, sr=44100, n_fft=4096, hop_length=1024, n_mels=512, device="cpu"):
	from torchaudio.transforms import AmplitudeToDB

	# Move tensors to GPU if available
	reference = torch.from_numpy(reference).float().to(device)
	estimate = torch.from_numpy(estimate).float().to(device)

	# Create a Hann window
	window = torch.hann_window(n_fft).to(device)

	# Compute STFTs with the Hann window
	D1 = torch.abs(torch.stft(reference, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True, pad_mode="constant"))
	D2 = torch.abs(torch.stft(estimate, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True, pad_mode="constant"))

	# create mel filterbank
	mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
	mel_filter_bank = torch.from_numpy(mel_basis).to(device)  # (melbandroformer is doing it that way) edit: sent to right device now

	# apply mel scale
	S1_mel = torch.matmul(mel_filter_bank, D1)
	S2_mel = torch.matmul(mel_filter_bank, D2)

	# Convert to decibels
	S1_db = AmplitudeToDB(stype="magnitude", top_db=80)(S1_mel)
	S2_db = AmplitudeToDB(stype="magnitude", top_db=80)(S2_mel)

	# Calculate difference
	diff = S2_db - S1_db

	# Separate positive and negative differences
	positive_diff = diff[diff > 0]
	negative_diff = diff[diff < 0]

	# Calculate averages
	average_positive = torch.mean(positive_diff) if positive_diff.numel() > 0 else torch.tensor(0.0).to(device)
	average_negative = torch.mean(negative_diff) if negative_diff.numel() > 0 else torch.tensor(0.0).to(device)

	# Scale with 100 as best score
	bleedless = 100 * 1 / (average_positive + 1)
	fullness = 100 * 1 / (-average_negative + 1)

	return bleedless.cpu().numpy(), fullness.cpu().numpy()


def get_metrics(
	metrics,
	reference,  # (ch, length)
	estimate,  # (ch, length)
	mix,  # (ch, length)
	device="cpu",
):
	result = dict()
	if "sdr" in metrics:
		references = np.expand_dims(reference, axis=0)
		estimates = np.expand_dims(estimate, axis=0)
		result["sdr"] = sdr(references, estimates)[0]
	if "si_sdr" in metrics:
		result["si_sdr"] = si_sdr(reference, estimate)
	if "l1_freq" in metrics:
		result["l1_freq"] = L1Freq_metric(reference, estimate, device=device)
	if "log_wmse" in metrics:
		result["log_wmse"] = LogWMSE_metric(reference, estimate, mix, device)
	if "aura_stft" in metrics:
		result["aura_stft"] = AuraSTFT_metric(reference, estimate, device)
	if "aura_mrstft" in metrics:
		result["aura_mrstft"] = AuraMRSTFT_metric(reference, estimate, device)
	if "bleedless" in metrics or "fullness" in metrics:
		bleedless, fullness = bleed_full(reference, estimate, device=device)
		if "bleedless" in metrics:
			result["bleedless"] = bleedless
		if "fullness" in metrics:
			result["fullness"] = fullness
	return result


# ============================================================================
# 原 webui/utils.py 的内容
# ============================================================================

# load and save config files
def load_configs(config_path):
	if config_path.endswith(".json"):
		with open(config_path, "r", encoding="utf-8") as f:
			return json.load(f)
	elif config_path.endswith(".yaml") or config_path.endswith(".yml"):
		with open(config_path, "r", encoding="utf-8") as f:
			return ConfigDict(yaml.load(f, Loader=yaml.FullLoader))


def save_configs(config, config_path):
	if config_path.endswith(".json"):
		with open(config_path, "w", encoding="utf-8") as f:
			json.dump(config, f, indent=4)
	elif config_path.endswith(".yaml") or config_path.endswith(".yml"):
		with open(config_path, "w", encoding="utf-8") as f:
			yaml.dump(config.to_dict(), f)


def color_config(config):
	def format_dict(d):
		items = []
		for k, v in sorted(d.items()):
			colored_key = f"\033[0;33m{k}\033[0m"
			if isinstance(v, dict):
				formatted_value = f"{{{format_dict(v)}}}"
			else:
				formatted_value = str(v)
			items.append(f"{colored_key}: {formatted_value}")
		return ", ".join(items)

	return f"{{{format_dict(config)}}}"


# get language from config file and setup i18n, model download main link
def get_language():
	try:
		config = load_configs(WEBUI_CONFIG)
		language = config["settings"].get("language", "Auto")
	except:
		language = "Auto"

	if language == "Auto":
		language = locale.getdefaultlocale()[0]
	return language


def get_main_link():
	try:
		config = load_configs(WEBUI_CONFIG)
		main_link = config["settings"]["download_link"]
	except:
		main_link = "Auto"

	if main_link == "Auto":
		main_link = "hf-mirror.com" if get_language() == "zh_CN" else "huggingface.co"
	return main_link


i18n = I18nAuto(get_language())


# webui restart function
def webui_restart():
	logger.info("Restarting WebUI...")
	os.execl(PYTHON, PYTHON, *sys.argv)


# setup webui debug mode
def log_level_debug(isdug):
	config = load_configs(WEBUI_CONFIG)
	if isdug:
		set_log_level(logger, logging.DEBUG)
		config["settings"]["debug"] = True
		save_configs(config, WEBUI_CONFIG)
		logger.info("Console log level set to \033[34mDEBUG\033[0m")
		return i18n("已开启调试日志")
	else:
		set_log_level(logger, logging.INFO)
		config["settings"]["debug"] = False
		save_configs(config, WEBUI_CONFIG)
		logger.info("Console log level set to \033[32mINFO\033[0m")
		return i18n("已关闭调试日志")


def load_selected_model(model_type=None):
	if not model_type:
		webui_config = load_configs(WEBUI_CONFIG)
		model_type = webui_config["inference"]["model_type"]
	if model_type:
		downloaded_model = []
		model_dir = os.path.join(MODEL_FOLDER, model_type)
		if not os.path.exists(model_dir):
			return None
		for files in os.listdir(model_dir):
			if files.endswith((".ckpt", ".th", ".chpt")):
				try:
					get_msst_model(files, model_type)
					downloaded_model.append(files)
				except:
					continue
		return downloaded_model
	return None


def load_msst_model():
	model_list = []
	model_classes = ["multi_stem_models", "single_stem_models", "vocal_models"]
	model_dir = [os.path.join(MODEL_FOLDER, keys) for keys in model_classes]
	for dirs in model_dir:
		for files in os.listdir(dirs):
			if files.endswith((".ckpt", ".th", ".chpt")):
				model_list.append(files)
	return model_list


def get_msst_model(model_name, model_type=None):
	config = load_configs(MODELS_INFO)
	main_link = get_main_link()
	model_type = [model_type] if model_type else ["multi_stem_models", "single_stem_models", "vocal_models"]
	if not model_name in config.keys():
		# print(model_name, config.keys())
		raise gr.Error(i18n("模型不存在!"))
	model = config[model_name]
	model_path = model["target_position"]
	config_path = model_path.replace("pretrain", "configs") + ".yaml"
	download_link = model["link"]
	model_type = model["model_type"]
	try:
		download_link = download_link.replace("huggingface.co", main_link)
	except:
		pass

	return model_path, config_path, model_type, download_link


def load_vr_model():
	downloaded_model = []
	config = load_configs(WEBUI_CONFIG)
	vr_model_path = config["settings"]["uvr_model_dir"]
	for files in os.listdir(vr_model_path):
		if files.endswith(".pth"):
			try:
				get_vr_model(files)
				downloaded_model.append(files)
			except:
				continue
	return downloaded_model


def get_vr_model(model):
	config = load_configs(MODELS_INFO)
	model_path = load_configs(WEBUI_CONFIG)["settings"]["uvr_model_dir"]
	main_link = get_main_link()

	for keys in config.keys():
		if keys == model:
			primary_stem = config[keys]["primary_stem"]
			secondary_stem = config[keys]["secondary_stem"]
			model_url = config[keys]["link"]
			try:
				model_url = model_url.replace("huggingface.co", main_link)
			except:
				pass
			return primary_stem, secondary_stem, model_url, model_path
	raise gr.Error(i18n("模型不存在!"))


# get model size and sha256 according to model name and model_info.json
def load_model_info(model_name):
	model_info = load_configs(MODELS_INFO)
	if model_name in model_info.keys():
		model_size = model_info[model_name].get("model_size", "Unknown")
		share256 = model_info[model_name].get("sha256", "Unknown")
		if model_size != "Unknown":
			model_size = round(int(model_size) / 1024 / 1024, 2)
	else:
		model_size = "Unknown"
		share256 = "Unknown"
	return model_size, share256


# update dropdown model list in webui according to selected model type
def update_model_name(model_type):
	if model_type == "UVR_VR_Models":
		model_map = load_vr_model()
		return gr.Dropdown(label=i18n("选择模型"), choices=model_map, interactive=True)
	else:
		model_map = load_selected_model(model_type)
		return gr.Dropdown(label=i18n("选择模型"), choices=model_map, interactive=True)


# change button visibility according to selected inference type
def change_to_audio_infer():
	return (gr.Button(i18n("输入音频分离"), variant="primary", visible=True), gr.Button(i18n("输入文件夹分离"), variant="primary", visible=False))


def change_to_folder_infer():
	return (gr.Button(i18n("输入音频分离"), variant="primary", visible=False), gr.Button(i18n("输入文件夹分离"), variant="primary", visible=True))


def select_folder():
	import tkinter as tk
	from tkinter import filedialog

	root = tk.Tk()
	root.withdraw()
	root.attributes("-topmost", True)
	selected_dir = filedialog.askdirectory()
	root.destroy()
	return selected_dir


def select_yaml_file():
	import tkinter as tk
	from tkinter import filedialog

	root = tk.Tk()
	root.withdraw()
	root.attributes("-topmost", True)
	selected_file = filedialog.askopenfilename(filetypes=[("YAML files", "*.yaml")])
	root.destroy()
	return selected_file


def select_file():
	import tkinter as tk
	from tkinter import filedialog

	root = tk.Tk()
	root.withdraw()
	root.attributes("-topmost", True)
	selected_file = filedialog.askopenfilename(filetypes=[("All files", "*.*")])
	root.destroy()
	return selected_file


def open_folder(folder):
	if folder == "":
		raise gr.Error(i18n("请先选择文件夹!"))
	os.makedirs(folder, exist_ok=True)
	absolute_path = os.path.abspath(folder)
	if platform.system() == "Windows":
		os.system(f"explorer {absolute_path}")
	elif platform.system() == "Darwin":
		os.system(f"open {absolute_path}")
	else:
		os.system(f"xdg-open {absolute_path}")


# error manager, add more detailed solutions according to the error message
def detailed_error(e):
	e = str(e)
	m = None

	if "CUDA out of memory" in e or "CUBLAS_STATUS_NOT_INITIALIZED" in e:
		m = i18n("显存不足, 请尝试减小batchsize值和chunksize值后重试。")
	elif "页面文件太小" in e or "DataLoader worker" in e or "DLL load failed while" in e or "[WinError 1455]" in e:
		m = i18n("内存不足，请尝试增大虚拟内存后重试。若分离时出现此报错，也可尝试将推理音频裁切短一些，分段分离。")
	elif "ffprobe not found" in e:
		m = i18n("FFmpeg未找到，请检查FFmpeg是否正确安装。若使用的是整合包，请重新安装。")
	elif "failed reading zip archive" in e:
		m = i18n("模型损坏，请重新下载并安装模型后重试。")
	elif "No such file or directory" in e or "系统找不到" in e or "[WinError 3]" in e or "[WinError 2]" in e or "The system cannot find the file specified" in e:
		m = i18n("文件或路径不存在，请根据错误指示检查是否存在该文件。")

	if m:
		e = m + "\n" + e
	return e


# ============================================================================
# 原 webui/init.py 的内容
# ============================================================================

def init_selected_model():
	try:
		batch_size, num_overlap, chunk_size, is_normalize = None, None, None, False
		config = load_configs(WEBUI_CONFIG)
		selected_model = config["inference"]["selected_model"]
		_, config_path, _, _ = get_msst_model(selected_model)
		config = load_configs(config_path)

		if config.inference.get("batch_size"):
			batch_size = int(config.inference.get("batch_size"))
		if config.inference.get("num_overlap"):
			num_overlap = int(config.inference.get("num_overlap"))
		if config.audio.get("chunk_size"):
			chunk_size = int(config.audio.get("chunk_size"))
		if config.inference.get("normalize"):
			is_normalize = True
		return batch_size, num_overlap, chunk_size, is_normalize
	except:
		return None, None, None, False


def init_selected_msst_model():
	webui_config = load_configs(WEBUI_CONFIG)
	selected_model = webui_config["inference"]["selected_model"]
	insts = [i18n("请先选择模型")]

	if not selected_model:
		return insts

	try:
		_, config_path, _, _ = get_msst_model(selected_model)
		config = load_configs(config_path)
		insts = config.training.instruments
		return insts
	except:
		return insts


def init_selected_vr_model():
	webui_config = load_configs(WEBUI_CONFIG)
	model = webui_config["inference"]["vr_select_model"]
	vr_primary_stem_only = i18n("仅输出主音轨")
	vr_secondary_stem_only = i18n("仅输出次音轨")

	if not model:
		return vr_primary_stem_only, vr_secondary_stem_only

	try:
		primary_stem, secondary_stem, _, _ = get_vr_model(model)
		vr_primary_stem_only = f"{primary_stem} Only"
		vr_secondary_stem_only = f"{secondary_stem} Only"
		return vr_primary_stem_only, vr_secondary_stem_only
	except:
		return vr_primary_stem_only, vr_secondary_stem_only