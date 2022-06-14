# Visual Acoustic Matching

## Motivation
We introduce the visual acoustic matching task, in which an audio clip is transformed to sound like it was recorded in a target environment. Given an image of the target environment and a waveform for the source audio, the goal is to re-synthesize the audio to match the target room acoustics as suggested by its visible geometry and materials. To address this novel task, we propose a cross-modal transformer model that uses audio-visual attention to inject visual properties into the audio and generate realistic audio output. In addition, we devise a self-supervised training objective that can learn acoustic matching from in-the-wild Web videos, despite their lack of acoustically mismatched audio. We demonstrate that our approach successfully translates human speech to a variety of real-world environments depicted in images, outperforming both traditional acoustic matching and more heavily supervised baselines.

<div style="text-align: center;"><img width="500"  src="https://changan.io/images/vam.png"></div>

## Installation 
Install this repo into pip by running the following command:
```
pip install -e .
```

## Usage
This repo supports benchmarking multiple visual acoustic research tasks (visual acoustic matching, 
audio-visual dereverberation and IR synthesis from images). We provide multiple baselines for comparison.


Below we provide instructions for evaluating the pretrained AViTAR models on SoundSpaces-Speech and AVSpeech.
To train the models from scratch, simply remove the ``--test --eval-best`` flags. 

1. Visual acoustic matching on SoundSpaces-Speech
```
python vam/trainer.py --model-dir data/pretrained-models/am --version avitar --model generative_avitar --batch-size 16 --num-encoder-layers 4 --n-gpus 1 --num-node 32 --use-rgb --use-depth --gpu-mem32 --acoustic-matching --use-cnn --pretrained-cnn --part learnlab --dropout 0 --log10 --remove-oov --decode-wav --hop-length 128 --auto-resume --slurm --encode-wav --use-visual-pe  --encoder-residual-layers 0 --decoder-residual-layers 0 --generator-lr 0.0005 --test --eval-best
```
2. Visual acoustic matching on Acoustic AVSpeech
```
py vam/trainer.py --model-dir data/pretrained-models/avspeech --version avitar --model generative_avitar --batch-size 16 --num-encoder-layers 4 --n-gpus 1 --num-node 32 --use-rgb --gpu-mem32 --acoustic-matching --use-cnn --pretrained-cnn --part learnlab --dropout 0 --log10 --remove-oov --decode-wav --hop-length 128 --auto-resume --slurm --encode-wav --use-visual-pe  --encoder-residual-layers 0 --decoder-residual-layers 0 --generator-lr 0.0005 --use-avspeech --num-worker 3 --dereverb-avspeech --convolve-random-rir --use-da --use-vida --use-audio-da --read-mp4 --adaptive-pool --test --eval-best
```
3. Inference
```
python vam/inference.py --model-dir data/pretrained-models/avspeech --version avitar --model generative_avitar --batch-size 16 --num-encoder-layers 4 --n-gpus 1 --num-node 32 --use-rgb --gpu-mem32 --acoustic-matching --use-cnn --pretrained-cnn --part learnlab --dropout 0 --log10 --remove-oov --decode-wav --hop-length 128 --auto-resume --slurm --encode-wav --use-visual-pe  --encoder-residual-layers 0 --decoder-residual-layers 0 --generator-lr 0.0005 --use-avspeech --num-worker 3 --dereverb-avspeech --convolve-random-rir --use-da --use-vida --use-audio-da --read-mp4 --adaptive-pool --test --eval-best
```

A couple useful flags (mostly work on SoundSpaces-Speech):
* ``--dereverb`` flag switches the model from acoustic matching to perform dereverberation
* ``--match-rir`` flag switches the model from acoustic matching to IR synthesis
* ``--visualize`` flag outputs qualitative examples


## Data
### SoundSpaces-Speech Dataset
Download the data from the links below and unzip them under ``data/soundspaces_speech`` directory
```angular2html
wget http://dl.fbaipublicfiles.com/SoundSpaces/av_dereverb/metadata.zip
wget http://dl.fbaipublicfiles.com/SoundSpaces/av_dereverb/val-mini.tar.xz
wget http://dl.fbaipublicfiles.com/SoundSpaces/av_dereverb/test-unseen.tar.xz
wget http://dl.fbaipublicfiles.com/SoundSpaces/av_dereverb/train.tar.xz
```

### AVSpeech Dataset
Please download [AVSpeech dataset](https://looking-to-listen.github.io/avspeech/) and follow our instructions in the paper to process data. More scripts will follow.


### Pretrained Model 
Run the command below to download pretrained weights
```
wget http://dl.fbaipublicfiles.com/vam/pretrained-models.zip
```

## Contributing
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.


## Citation
If you find the code, data, or models useful for your research, please consider citing the following two papers:
```
@inproceedings{chen22vam,
  title     =     {Visual Acoustic Matching,
  author    =     {Changan Chen and Ruohan Gao and Paul Calamia and Kristen Grauman},
  booktitle =     {CVPR},
  year      =     {2022}
}
```
```
@inproceedings{chen22dereverb,
  title     =     {Learning Audio-Visual Dereverberation,
  author    =     {Changan Chen and Wei Sun and David Harwath and Kristen Grauman},
  booktitle =     {arxiv},
  year      =     {2022}
}
```

## License
This repo is CC-BY-NC licensed, as found in the [LICENSE](LICENSE) file.