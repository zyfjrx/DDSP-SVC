import librosa
import yaml
import os

from tools.SOME import inference
from tools.SOME.utils.infer_utils import build_midi_file
from tools.SOME.utils.slicer2 import Slicer
from logger.saver import Saver

# 创建一个简单的logger类
class SimpleLogger:
    def info(self, msg):
        print(f"[INFO] {msg}")
    
    def error(self, msg):
        print(f"[ERROR] {msg}")

logger = SimpleLogger()
# pretrained SOME weight
SOME_WEIGHT = "tools/SOME_weights/model_steps_64000_simplified.ckpt"
SOME_CONFIG = "configs/config_some.yaml"

def infer(model_path, config_path, wav_path, output_dir, tempo):
	with open(config_path, "r", encoding="utf8") as f:
		config = yaml.safe_load(f)
	infer_cls = inference.task_inference_mapping[config["task_cls"]]

	if infer_cls == "MIDIExtractionInference":
		from tools.SOME.inference.me_infer import MIDIExtractionInference

		infer_ins = MIDIExtractionInference(config=config, model_path=model_path)
	elif infer_cls == "QuantizedMIDIExtractionInference":
		from tools.SOME.inference.me_quant_infer import QuantizedMIDIExtractionInference

		infer_ins = QuantizedMIDIExtractionInference(config=config, model_path=model_path)
	else:
		raise ValueError(f"Unknown inference class: {infer_cls}")

	waveform, _ = librosa.load(wav_path, sr=config["audio_sample_rate"], mono=True)
	slicer = Slicer(sr=config["audio_sample_rate"], max_sil_kept=1000)
	chunks = slicer.slice(waveform)
	midis = infer_ins.infer([c["waveform"] for c in chunks])
	midi_file = build_midi_file([c["offset"] for c in chunks], midis, tempo=tempo)

	os.makedirs(output_dir, exist_ok=True)
	wav_name = os.path.splitext(os.path.basename(wav_path))[0]
	midi_path = os.path.join(output_dir, f"{wav_name}.mid")
	midi_file.save(midi_path)

	return midi_path

def some_inference(audio_file, bpm, output_dir):
	if not os.path.isfile(SOME_WEIGHT):
		return ("请先下载SOME预处理模型并放置在tools/SOME_weights文件夹下! ")

	os.makedirs(output_dir, exist_ok=True)

	tempo = float(bpm)
	try:
		logger.info(f"Running SOME inference with audio file: {audio_file}, output dir: {output_dir}, tempo: {tempo}")
		midi = infer(SOME_WEIGHT, SOME_CONFIG, audio_file, output_dir, tempo)
		logger.info(f"SOME inference completed, MIDI file saved as: {midi}")
		return ("处理完成, 文件已保存为: ") + midi
	except Exception as e:
		logger.error(f"Fail to run SOME inference. Error: {e}\n{traceback.format_exc()}")
		return ("处理失败!") + str(e)