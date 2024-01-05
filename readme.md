# 🍅+🍎 = fuyu

here are some simple scripts to see if fuyu-8b works with your resources.
the mock dataset makes fake sentences and fake images of a size since the processor working with this model can have various issues (e.g. the tokens not being in vocab or image patches not being correctly put together).

to run you need: `torch`, `transformers`, `simple_parsing`.

# `train-simple.py`
this is just training without any extra bells + whistles

run with `python train-simple.py`

# `accelerate-train.py`
this uses accelerate with deepspeed.
as part of the GPU poor, I chop the model with `model.language_model.model.layers = model.language_model.model.layers[:2]` after the model is loaded to ensure the script works.

to run use: `accelerate launch accelerate-train.py`
this requires installing `accelerate` and **DOES NOT** use ~/.cache/huggingface/accelerate/default_config.yaml
