# Обучение
Репозиторий с проектом по теме Speech enhancement


```bash
conda activate universe

# train the model (UNIVERSE++, Voicebank-DEMAND, 16 kHz)
python3 models/universe/train.py experiment=universepp_vb_16k


# inference (first needs to move last.ckpt into .hydra folder)
python -m open_universe.bin.enhance \
		--model exp/universepp_vb_16k/2025-03-06_23-30-48_/.hydra/last.ckpt \
		data/voicebank_demand/16k/test/noisy \
		results/voicebank_16k/universe_pp_manual/

# test results
python3 -m open_universe.bin.eval_metrics results/voicebank_16k/universe_pp_manual/ \
  --ref_path data/voicebank_demand/16k/test/clean/ \
  --metrics dnsmos lps lsd pesq-wb si-sdr stoi-ext \
  --result_dir results/voicebank_16k/universe_pp_manual/

```
