# personality_edit

## Run

### Training

* Mend training

You can run follow commands:

```bash
python edit.py \
	--editing_method MEND_train \
	--hparams_path ./config/llama-7b-MEND.yaml \
	--data_dir ./data
```

You can change the `hparams_path` to `llama-7b.yaml` for editing llama models, and modify the detail parameters in `.yaml` files.

### Testing

* Mend testing

After training , you can set the `archive` in the specific config file to your trained model for evaluation, and the run the following command:

```bash
python edit.py \
	--editing_method MEND_test \
	--hparams_path ./config/llama-7b-MEND.yaml \
	--data_dir ./data \
	--TPEI \
	--PAE \
	--cls_path ./models/per-classifier \
	--metric_file ./MEND_test_metrics.json
```

* IKE testing

```bash
python edit.py \
	--editing_method IKE \
	--hparams_path ./config/personality/llama3-8b-ike.yaml \
	--data_dir ./data \
	--TPEI \
	--PAE \
	--cls_path ./models/per-classifier \
	--metric_file PATH_TO_METRIC_file
```



## Run

### Training

* Mend training

You can run follow commands:

```bash
python edit.py \
	--editing_method MEND_train \
	--hparams_path ./config/train/mend/gpt-j-6B.yaml \
	--data_dir ./data
```

You can change the `hparams_path` to `llama-7b.yaml` for editing llama models, and modify the detail parameters in `.yaml` files.

### Testing

* Mend testing

After training , you can set the `archive` in the specific config file to your trained model for evaluation, and the run the following command:

```bash
python edit.py \
	--editing_method MEND_test \
	--hparams_path ./config/test/mend/gpt-j-6B.yaml \
	--data_dir ./data \
	--TPEI \
	--PAE \
	--cls_path ./models/per-classifier \
	--metric_file PATH_TO_METRIC_file
```

* IKE testing

```bash
python edit.py \
	--editing_method IKE \
	--hparams_path ./config/test/ike/gpt-j-6B.yaml \
	--data_dir ./data \
	--TPEI \ 
	--PAE \ 
	--cls_path ./models/per-classifier \ 
	--metric_file PATH_TO_METRIC_file 
```