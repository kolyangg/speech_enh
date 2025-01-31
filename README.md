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

### universe / universe++
```bash
conda activate universe

python3 -m models/universe/open_universe.bin.enhance datasets/voicebank_demand/16k/test/noisy results/voicebank_16k/universe_pp/ \
  --model line-corporation/open-universe:original

python3 -m models/universe/open_universe.bin.enhance datasets/voicebank_demand/16k/test/noisy results/voicebank_16k/universe_pp/ \
  --model line-corporation/open-universe:plusplus
```
