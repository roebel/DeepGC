# Beyond Voice Identity Conversion: Manipulating Voice Attributes by Adversarial Learning of Structured Disentangled Representations

## Dependencies

* numpy
* tensorflow 2.2
* librosa
* pysndfile
* matplotlib

## Data
The [VCTK dataset](https://datashare.ed.ac.uk/handle/10283/3443) using mic1 only and discarding speakers p315 and s5 (108 speakers).

## Gender Converter
Go to *gender_converter* directory:
```bash
$ cd gender_converter
```

## Usage

### Installation
Install Python dependencies.
```bash
$ pip install -r requirements.txt
```

### Feature Extraction

* You need to extract 
    - log-mel-spectrograms from VCTK audios, using *calc_melspec.py*.
    - Then you need to compute mean and standard deviation of the mel-spectrograms on the entire training database using *compute_mean_std.py*,
    - You also need to run *compute_mean_std_spkencoder.py* using the VC model (see steps below) - for standardization of the output of the speaker encoder.
* You also need to do a forced alignement at phone level on all the VCTK sentences.

### Customize data reader

Write a snippet of code to walk through the dataset for generating list file for train, valid and test set.

Then you will need to modify the data reader to read your training data. The following are scripts you will need to modify:

- reader.py
- symbols.py


### Pre-train the VC model
```bash
$ python train_vc.py -o outdir_vc_model_80ep --hparams=epochs=80,batch_size=32 --db_root_dir pathtovctk
```
where **pathtovctk** is the path to your VCTK data AND lists, 
*outdir_vc_model_80ep* will contain the output VC model (and logdir for tensorboard).

### Compute statistics on speaker encoder output using training database
*  If you trained the VC model by your self:
    ```bash
    $ python compute_mean_std_spkencoder.py --model outdir_vc_model_80ep --db_root_dir pathtovctk --ckpt
    ```
* OR *using pre-trained models*:
    ```bash
    $ python compute_mean_std_spkencoder.py --model pathtomodels/model_vc_80ep_w/vc_weights.tf --db_root_dir pathtovctk
    ```

### Pre-train the "pre-trained gender discriminator" model
```bash
$ python train_gender_discriminator.py -i outdir_vc_model_80ep --db_root_dir pathtovctk -o outdir_genderdiscriminator
```
where *outdir_genderdiscriminator* will contain the output gender dicriminator model.

### Train the "gender auto-encoder" model
```bash
$ python train_aegender.py -igender outdir_genderdiscriminator -ivc outdir_vc_model_80ep --hparams=epochs=400,batch_size=64,gender_autoencoder_error_type=mae,gender_latent_dim=60,learning_rate=1e-4 --db_root_dir pathtovctk -o outdir_genderae_latentdim60
```
where *outdir_genderae_latentdim60* will contain the output gender model.
### Inference
Run the inference codes to generate audio samples. 
1. generate file lists for inference: 
    ```bash 
    $ python make_inference_lists.py --db_root_dir pathtovctk --suffix eval --dir mylist --set valid -n 3 p232 p274 p300 p253
    ```
2. Inference with models:
    * Inference with pre-trained models: 
      ```bash 
      python inference_cpu_list_gender.py --vcmodel pathtomodels/model_vc_80ep_w/vc_weights.tf --gendermodel pathtomodels/model_genderconversion_latentdim60_ep400_w/gender_weights.tf --mellist mylist/mel_eval.list --phonelist mylist/phone_eval.list --db_root_dir pathtovctk --out genderconversion_latentdim60_with_weights
      ```
      where **pathtomodels** is the path to pre-trained models.
      The folder **vc_genderconversion_latentdim60_with_weights** will contain the infered log-mel-spectrograms. They can be inverted using Griffin-Lim or better using WaveGlow.
    * Inference with computed models with the training commands above: 
      ```bash 
      python inference_cpu_list_gender.py --mellist mylist/mel_eval.list --phonelist mylist/phone_eval.list --task vc --vcmodel outdir_vc_model_80ep --gendermodel outdir_genderae_latentdim60 --db_root_dir pathtovctk --out genderconversion_latentdim60 --ckpt
      ```
      **vc_genderconversion_latentdim60** will contain the infered log-mel-spectrograms. 
3. The log-mel-spectrograms can be inverted using WaveGlow (see below) OR using with the *inference_cpu_list_gender.py* above command with --wav option (Griffin-Lim)


## Training Time
All training has been run on a single GPU (GForce GFX
1080Ti).
The
duration of the VC model training is 20 minutes per epoch
with 80 epochs (roughly 27 hours) and the training of the
gender autoencoder model lasts 1 minute and 30 seconds per
epoch with 400 epochs (total of 10 hours).

## WaveGlow Mel Inverter

### run WaveGlow Mel Inverter on mel files
```bash
../waveglow_mel_inverter/resynth_mel.py -i input_mel_files -o output_dir
```


#### Get Options
```bash
../waveglow_mel_inverter/resynth_mel.py -h
```

## Acknowledgements

Part of the gender conversion code was adapted from the project: https://github.com/jxzhanggg/nonparaSeq2seqVC_code
