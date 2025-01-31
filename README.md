# Speech enhancement
Репозиторий с проектом по теме Speech enhancement


# Модели
 
За benchmark архитектуры были взяты модели:
- Diffwave (https://github.com/lmnt-com/diffwave)
- Wavegrad (https://github.com/lmnt-com/wavegrad)
- HiFi-GAN (https://github.com/jik876/hifi-gan)
- HiFi++ (https://github.com/SamsungLabs/hifi_plusplus)
- Universe / Universe++ (https://github.com/line/open-universe)
 
# Использование кода

```bash
cd speech_enh
```

## 1. Настройка окружения всех моделей + скачивание чекпоинтов

```bash
models/diffwave/setup.sh
models/wavegrad/setup.sh
models/hifi-gan/setup.sh
models/hifi_pp/setup.sh
models/universe/setup.sh
```


## 2. Скачивание и подготовка данных
### Voicebank Demand датасет
```bash
conda activate universe
models/universe/data/prepare_voicebank_demand.sh
```

## 3. Инференс

### diffwave
```bash
conda activate diffwave
python3 -m models/diffwave/diffwave.preprocess datasets/voicebank_demand/16k/test/noisy
models/diffwave/diff_inference.sh diffwave-ljspeech-22kHz-1000578.pt datasets/voicebank_demand/16k/test/noisy results/voicebank_16k/diffwave
```

### wavegrad
```bash
conda activate diffwave # works for wavegrad
python3 -m models/wavegrad/wavegrad.preprocess datasets/voicebank_demand/16k/test/noisy
models/wavegrad/wave_inference.sh wavegrad-24kHz.pt datasets/voicebank_demand/16k/test/noisy results/voicebank_16k/wavegrad
```

### hifi-gan
```bash
conda activate hifigan

python3 models/hifigan/inference_NS.py --checkpoint_file checkpoints/hifigan/generator_v3 \
--input_wavs_dir datasets/voicebank_demand/16k/test/noisy \
--output_dir results/voicebank_16k/hifi-gan
```

### hifi_pp
```bash
conda activate universe # works for hifi_pp

WANDB_MODE=disabled python3  main.py exp.config_dir=models/hifi_pp/configs exp.config=models/hifi_pp/denoising.yaml exp.name="wtf" \
data.dir4inference=data/voicebank_demand/16k/test/noisy/hifi_pp checkpoint.checkpoint4inference=checkpoints/hifi_pp/se.pth \
data.voicebank_dir=data/voicebank_demand/16k/test/
```


### universe / universe++
```bash
conda activate universe

python3 -m models/universe/open_universe.bin.enhance datasets/voicebank_demand/16k/test/noisy results/voicebank_16k/universe_pp/ \
  --model line-corporation/open-universe:original

python3 -m models/universe/open_universe.bin.enhance datasets/voicebank_demand/16k/test/noisy results/voicebank_16k/universe_pp/ \
  --model line-corporation/open-universe:plusplus
```
