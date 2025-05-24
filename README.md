# Speech enhancement
Репозиторий с проектом по теме Speech enhancement



# Использование кода

```bash
git clone --recursive https://github.com/kolyangg/speech_enh.git

cd speech_enh
```


# I. Инференс улучшенных версий модели (на основе UNIVERSE++)

## 1. Настройка окружения + скачивание датасета Voicebank + DEMAND (происходит параллельно установке окружения)

```bash
models/universe/serv_setup.sh --mamba
## введите ключ для wandb при необходимости
## выберите y/n для последующего создания MFA меток для датасета
source activate
conda activate universe
# в случае ошибки запустить скрипт ещё раз
```

## 2. Скачивание чекпоинтов финальных моделей (Miipher + Double, Simple + Large, Original)

```bash
models/universe/utils/checkp_dl.sh
```

## 3. Скачивание сложного датасета

```bash
 models/universe/utils/hard_ds_dl.sh
```

## 3. Инференс на сложном датасете
```bash
export PYTHONPATH=$PWD/models/universe:$PYTHONPATH

# Miipher + Double
python3 models/universe/open_universe/bin/enhance_NS.py exp2/hard_dataset/all/noisy results/hard_ds/miiph+dbl --model exp2/wv_double+ff_m2_wvloss_18May/checkpoints/universe/exper/last.ckpt --text-path exp2/hard_dataset/transcripts

# Simple + Large
python3 models/universe/open_universe/bin/enhance_NS.py exp2/hard_dataset/all/noisy results/hard_ds/simpl_l --model exp2/16May_full_film_xph_wv/checkpoints/universe/exper/last.ckpt --text-path exp2/hard_dataset/transcripts

# Original (50k steps) - no text
python3 models/universe/open_universe/bin/enhance_NS.py exp2/hard_dataset/all/noisy results/hard_ds/orig --model exp2/orig_bucket_cluster/checkpoints/universe/exper/last.ckpt
```

## 4. Подсчёт метрик
```bash
# Miipher + Double
python3 -m open_universe.bin.eval_metrics  results/hard_ds/miiph+dbl \
  --ref_path exp2/hard_dataset/all/clean \
  --metrics dnsmos lps lsd pesq-wb si-sdr stoi-ext \
  --result_dir metrics/hard_dataset/miiph+dbl/ 

# Simple + Large

python3 -m open_universe.bin.eval_metrics results/hard_ds/simpl_l \
  --ref_path exp2/hard_dataset/all/clean \
  --metrics dnsmos lps lsd pesq-wb si-sdr stoi-ext \
  --result_dir metrics/hard_dataset/simpl_l/ 

# Original (50k steps)
python3 -m open_universe.bin.eval_metrics results/hard_ds/orig \
  --ref_path exp2/hard_dataset/all/clean \
  --metrics dnsmos lps lsd pesq-wb si-sdr stoi-ext \
  --result_dir metrics/hard_dataset/orig/ 
```

# II. Бенчмарк чекпоинтов базовых моделей

## Базовые модели
 
За benchmark архитектуры были взяты модели:
- Diffwave (https://github.com/lmnt-com/diffwave)
- Wavegrad (https://github.com/lmnt-com/wavegrad)
- HiFi-GAN (https://github.com/jik876/hifi-gan)
- HiFi++ (https://github.com/SamsungLabs/hifi_plusplus)
- Universe / Universe++ (https://github.com/line/open-universe)
 

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
python3 -m diffwave.preprocess data/voicebank_demand/16k/test/noisy
models/diffwave/diff_inference.sh checkpoints/diffwave/diffwave-ljspeech-22kHz-1000578.pt data/voicebank_demand/16k/test/noisy results/voicebank_16k/diffwave
```

### wavegrad
```bash
conda activate diffwave # works for wavegrad
python3 -m wavegrad.preprocess data/voicebank_demand/16k/test/noisy
models/wavegrad/wave_inference.sh checkpoints/wavegrad/wavegrad-24kHz.pt data/voicebank_demand/16k/test/noisy results/voicebank_16k/wavegrad
```

### hifi-gan
```bash
conda activate hifigan

python3 models/hifi-gan/inference_NS.py --checkpoint_file checkpoints/hifigan/generator_v3.pt \
--input_wavs_dir data/voicebank_demand/16k/test/noisy \
--output_dir results/voicebank_16k/hifi-gan
```

### hifi_pp
```bash
conda activate universe # works for hifi_pp
rm data/voicebank_demand/16k/test/noisy/*.spec.npy # remove npy files

WANDB_MODE=disabled python3  models/hifi_pp/main.py exp.config_dir=models/hifi_pp/configs exp.config="denoising.yaml" exp.name="wtf" \
data.dir4inference=results/voicebank_16k/hifi_pp checkpoint.checkpoint4inference=checkpoints/hifi_pp/se.pth \
data.voicebank_dir=data/voicebank_demand/16k/test/
```


### universe / universe++
```bash
conda activate universe

python3 -m open_universe.bin.enhance data/voicebank_demand/16k/test/noisy results/voicebank_16k/universe_orig/ \
  --model line-corporation/open-universe:original

python3 -m open_universe.bin.enhance data/voicebank_demand/16k/test/noisy results/voicebank_16k/universe_pp/ \
  --model line-corporation/open-universe:plusplus
```

## 4. Подсчёт метрик
```bash
conda activate universe

# diffwave
python3 -m open_universe.bin.eval_metrics results/voicebank_16k/diffwave  \
  --ref_path data/voicebank_demand/16k/test/clean/ \
  --metrics dnsmos lps lsd pesq-wb si-sdr stoi-ext \
  --result_dir metrics/voicebank_16k/diffwave/ 

# wavegrad
python3 -m open_universe.bin.eval_metrics results/voicebank_16k/wavegrad \
  --ref_path data/voicebank_demand/16k/test/clean/ \
  --metrics dnsmos lps lsd pesq-wb si-sdr stoi-ext \
  --result_dir metrics/voicebank_16k/wavegrad/ 

# hifi-gan
python3 -m open_universe.bin.eval_metrics results/voicebank_16k/hifi-gan \
  --ref_path data/voicebank_demand/16k/test/clean/ \
  --metrics dnsmos lps lsd pesq-wb si-sdr stoi-ext \
  --result_dir metrics/voicebank_16k/hifi-gan/ 

# hifi++
python3 -m open_universe.bin.eval_metrics results/voicebank_16k/hifi_pp \
  --ref_path data/voicebank_demand/16k/test/clean/ \
  --metrics dnsmos lps lsd pesq-wb si-sdr stoi-ext \
  --result_dir metrics/voicebank_16k/hifi_pp/ 

# universe orig
python3 -m open_universe.bin.eval_metrics results/voicebank_16k/universe_orig/ \
  --ref_path data/voicebank_demand/16k/test/clean \
  --metrics dnsmos lps lsd pesq-wb si-sdr stoi-ext \
  --result_dir metrics/voicebank_16k/universe_orig/ 

# universe++
python3 -m open_universe.bin.eval_metrics results/voicebank_16k/universe_pp/  \
  --ref_path data/voicebank_demand/16k/test/clean/ \
  --metrics dnsmos lps lsd pesq-wb si-sdr stoi-ext \
  --result_dir metrics/voicebank_16k/universe_pp/ 

```

## 5. Таблица с результатами

```bash
conda activate universe

python3 -m open_universe.bin.make_table \
    --format github \
    --results   metrics/voicebank_16k/diffwave/diffwave_summary.json \
                metrics/voicebank_16k/wavegrad/wavegrad_summary.json \
                metrics/voicebank_16k/hifi-gan/hifi-gan_summary.json \
                metrics/voicebank_16k/hifi_pp/hifi_pp_summary.json \
                metrics/voicebank_16k/universe_orig/universe_orig_summary.json \
                metrics/voicebank_16k/universe_pp/universe_pp_summary.json \
    --labels DiffWave WaveGrad Hifi_GAN HiFi++ UNIVERSE UNIVERSE++
```

## 6. Результаты разных моделей на датасете Voicebank Demand (16k)

| model      |   si-sdr |   pesq-wb |   stoi-ext |    lsd |   lps |   OVRL |   SIG |   BAK |
|------------|----------|-----------|------------|--------|-------|--------|-------|-------|
| DiffWave   |  -43.039 |     1.082 |      0.457 |  9.959 | 0.799 |  2.444 | 3.002 | 3.178 |
| WaveGrad   |  -28.470 |     1.124 |      0.518 | 21.592 | 0.731 |  2.043 | 2.767 | 2.238 |
| Hifi_GAN   |  -40.591 |     1.055 |      0.155 | 13.404 | 0.423 |  2.363 | 3.015 | 2.809 |
| HiFi++     |  -19.309 |     2.301 |      0.319 | 13.471 | 0.854 |  3.008 | 3.329 | 3.888 |
| UNIVERSE   |   17.596 |     2.835 |      0.844 |  6.320 | 0.921 |  3.158 | 3.457 | 4.013 |
| UNIVERSE++ |   18.630 |     3.013 |      0.864 |  4.868 | 0.935 |  3.204 | 3.492 | 4.043 |

## 7. Скачивание сложных семплов
```bash
python3 src/hard_dataset.py

```

## 8. Инференс и подсчёт метрик

```bash
python3 -m open_universe.bin.enhance data/hard_dataset/paired/noisy results/hard_dataset/universe_pp/paired \
  --model line-corporation/open-universe:plusplus
python3 -m open_universe.bin.enhance data/hard_dataset/all/noisy results/hard_dataset/universe_pp/all/ \
  --model line-corporation/open-universe:plusplus
python3 -m open_universe.bin.enhance data/hard_dataset/cv/noisy results/hard_dataset/universe_pp/cv/ \
  --model line-corporation/open-universe:plusplus
python3 -m open_universe.bin.enhance data/hard_dataset/vox/noisy results/hard_dataset/universe_pp/vox/ \
  --model line-corporation/open-universe:plusplus

```

```bash
conda activate universe

python3 -m open_universe.bin.eval_metrics results/hard_dataset/universe_pp/paired  \
  --ref_path data/hard_dataset/paired/clean \
  --metrics dnsmos lps lsd pesq-wb si-sdr stoi-ext \
  --result_dir metrics/hard_dataset/universe_pp/paired
python3 -m open_universe.bin.eval_metrics results/hard_dataset/universe_pp/all  \
  --metrics dnsmos lps lsd pesq-wb si-sdr stoi-ext \
  --result_dir metrics/hard_dataset/universe_pp/all
python3 -m open_universe.bin.eval_metrics results/hard_dataset/universe_pp/cv  \
  --metrics dnsmos lps lsd pesq-wb si-sdr stoi-ext \
  --result_dir metrics/hard_dataset/universe_pp/cv
python3 -m open_universe.bin.eval_metrics results/hard_dataset/universe_pp/vox  \
  --metrics dnsmos lps lsd pesq-wb si-sdr stoi-ext \
  --result_dir metrics/hard_dataset/universe_pp/vox
```

```bash
conda activate universe

python3 -m open_universe.bin.make_table \
    --format github \
    --results   metrics/hard_dataset/universe_pp/paired/paired_summary.json \
                metrics/hard_dataset/universe_pp/all/all_summary.json \
                metrics/hard_dataset/universe_pp/cv/cv_summary.json \
                metrics/hard_dataset/universe_pp/vox/vox_summary.json \
    --labels Paired All CV Vox
```

## 9. Результаты Universe++ на сложных семплах

| model   |   si-sdr |   pesq-wb |   stoi-ext |    lsd |   lps |   OVRL |   SIG |   BAK |
|---------|----------|-----------|------------|--------|-------|--------|-------|-------|
| Paired  |  -27.111 |     1.060 |      0.155 | 21.935 | 0.148 |  1.958 | 2.225 | 3.509 |
| All     |          |           |            |        |       |  2.597 | 2.914 | 3.787 |
| CV      |          |           |            |        |       |  2.406 | 2.754 | 3.674 |
| Vox     |          |           |            |        |       |  3.172 | 3.457 | 4.023 |