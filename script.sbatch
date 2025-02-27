#!/bin/bash
#SBATCH --job-name=aboba             # Заместо aboba напишите ваше незвание эксперимента
#SBATCH --error=runs/aboba-%j.err    # Файл для вывода ошибок
#SBATCH --output=runs/aboba-%j.log   # Файл для вывода результатов
#SBATCH --gpus=2                     # Количество запрашиваемых гпу
#SBATCH --cpus-per-task=16            # Выполнение расчёта на 8 ядрах CPU
#SBATCH --time=5:00:00               # Максимальное время выполнения (чч:мм:сс), после его окончания програмаа просто сбрасывается
#SBATCH --constraint="[type_a|type_b|type_c]"   # Выбор, какие узлы использовать. 
																		 # Узлы a, b и c - содержат v100, d без гпу, e - содержит a100, простым смертным недоступен (нужно писать Айбеку)

module load Python/Anaconda_v03.2023

# source deactivate
source deactivate

source activate universe
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH 

# python3 -m open_universe.bin.enhance data/voicebank_demand/16k/test/noisy results/voicebank_16k/universe_pp/ \
#   --model line-corporation/open-universe:plusplus

python3 models/universe/train.py experiment=universepp_vb_16k
