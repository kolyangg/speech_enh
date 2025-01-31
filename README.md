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

```


## 2. Скачивание и подготовка данных
# Voicebank Demand датасет
```bash
models/universe/data/prepare_voicebank_demand.sh
```

## 3. Инференс

### diffwave
```bash
conda activate diffwave
models/diffwave/diff_inference.sh diffwave-ljspeech-22kHz-1000578.pt results/voicebank_16k/diffwave
```

