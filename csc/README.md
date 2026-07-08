# Kuuntelija CSC:n Puhtilla (eräajo)

Ohjeet `kuuntelija30b.py`:n ajamiseen Slurm-eräajona Puhtin GPU-noodeilla.
Puhtin V100:ssa on 32 Gt muistia, joten koko Q4-malli mahtuu näytönohjaimeen
(`KUUNTELIJA_CPU_MOE=0`). Sama toimii Mahtilla (A100 40 Gt) partitiota
vaihtamalla.

> Tämä ohje on käyty läpi Puhtilla (release 33, heinäkuu 2026) ja sisältää
> kohdatut sudenkuopat. Kustannus ~60 BU / GPU-tunti, eli ~400 BU tuhannen
> biisin kansiolle. **Puhti suljetaan ~kuukausi Roihun GA:n jälkeen (kesä
> 2026)** — Allakseen viety data ja tämä ohje siirtyvät Roihulle, vain
> partitio/GPU-tyyppi vaihtuu.

## Kertavalmistelut (kirjautumisnoodilla)

Kaikki lataukset tehdään kirjautumisnoodilla — **laskentanoodeilla ei ole
internet-yhteyttä**. Aja komennot yksi kerrallaan (Puhtin selainpääte
sotkee monirivisen liitoksen).

### 0. Muuttujat ja repo

```sh
export PROJEKTI=project_XXXXXXX          # oma projektinumerosi
export TYOTILA=/scratch/$PROJEKTI/kuuntelija
mkdir -p $TYOTILA && cd $TYOTILA
git clone https://github.com/pekkaouli/kuuntelija.git .
```

### 1. Mallit (Xet-siirto — älä käytä curlia)

HF:n tavallinen CDN throttlaa jaettua CSC-IP:tä rajusti (~300 kB/s) ja
`curl`/`wget` törmää HTTP/2-katkoksiin. HF:n Xet-siirto on ~1000× nopeampi
(mitattu 330 MB/s):

```sh
module load python-data
pip install --user huggingface_hub hf_transfer hf_xet
mkdir -p $TYOTILA/mallit && cd $TYOTILA/mallit
HF_XET_HIGH_PERFORMANCE=1 python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF', filename='Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf', local_dir='.')"
HF_XET_HIGH_PERFORMANCE=1 python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF', filename='mmproj-Qwen3-Omni-30B-A3B-Instruct-bf16.gguf', local_dir='.')"
cd $TYOTILA
```

> **Tärkeä siivous:** `pip install --user huggingface_hub` asentaa uuden
> version (1.x) `~/.local`:iin, joka myöhemmin **rikkoo pytorch-moduulin
> transformersin** (vaatii `huggingface-hub<1.0`). Poista se heti mallien
> latauksen jälkeen — mallit on jo ladattu, sitä ei enää tarvita:
> ```sh
> rm -rf ~/.local/lib/python*/site-packages/huggingface_hub ~/.local/lib/python*/site-packages/huggingface_hub-*.dist-info
> ```

### 2. llama.cpp CUDA-tuella

Kirjautumisnoodilla ei ole GPU-ajuria (`libcuda.so.1`), joten linkkaus
kaatuu ilman ajurin stub-kirjastoa. Toolkit sisältää stubin nimellä
`libcuda.so`, mutta linkkeri etsii versioitua `libcuda.so.1` — tehdään
symlink ja osoitetaan linkkeri siihen (stub on vain käännösaikaa varten,
ajossa GPU-noodin oikea ajuri löytyy):

```sh
module load gcc cuda cmake
git clone https://github.com/ggml-org/llama.cpp

CUDA_ROOT=$(dirname $(dirname $(which nvcc)))
mkdir -p $TYOTILA/cudastub
ln -sf $CUDA_ROOT/lib64/stubs/libcuda.so $TYOTILA/cudastub/libcuda.so.1

cmake -S $TYOTILA/llama.cpp -B $TYOTILA/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_EXE_LINKER_FLAGS="-L$TYOTILA/cudastub -Wl,-rpath-link,$TYOTILA/cudastub" -DCMAKE_SHARED_LINKER_FLAGS="-L$TYOTILA/cudastub -Wl,-rpath-link,$TYOTILA/cudastub"
cmake --build $TYOTILA/llama.cpp/build --config Release -j 8 --target llama-mtmd-cli
```

(Mahtin A100:lle `-DCMAKE_CUDA_ARCHITECTURES=80`.)

### 3. Python-ympäristö

`python-data`-moduulissa ei ole torchia; käytä `pytorch`-moduulia ja luo
sen päälle venv, joka perii torchin ja transformersin ja lisää vain
audio-kirjastot:

```sh
module purge
module load pytorch
python3 -m venv --system-site-packages $TYOTILA/.venv
$TYOTILA/.venv/bin/pip install librosa soundfile
$TYOTILA/.venv/bin/python -c "import torch,transformers,librosa,soundfile,numpy; print(torch.__version__, transformers.__version__, librosa.__version__)"
```

Viimeinen rivi tulostaa versiot, jos kaikki on kunnossa. Jos se valittaa
`huggingface-hub==1.x`, teit kohdan 1 siivouksen tekemättä — poista
`~/.local`:n huggingface_hub yllä olevalla komennolla.

### 4. Esilataa HF-luokittelijat välimuistiin

Laskentanoodi ei voi ladata näitä, joten haetaan ne nyt:

```sh
export HF_HOME=$TYOTILA/hf-cache
export HF_HUB_ENABLE_HF_TRANSFER=1
$TYOTILA/.venv/bin/python -c "from transformers import pipeline; pipeline('audio-classification', model='dima806/music_genres_classification', device=-1); pipeline('audio-classification', model='MIT/ast-finetuned-audioset-10-10-0.4593', device=-1); print('valmis')"
```

### 5. Musiikki työtilaan

Allaksesta (jos konfiguroit rclone-remoten MyCSC:n Cloud storage
-paneelista) tai suoraan omalta koneelta:

```sh
mkdir -p $TYOTILA/musiikki
rclone copy s3allas-project_XXXXXXX:BUCKETTI $TYOTILA/musiikki -P
# tai omalta koneelta: rsync -av ~/Musiikki/ puhti.csc.fi:$TYOTILA/musiikki/
```

## Ajo

Projektinumero annetaan `sbatch`ille (skripteissä on placeholder, jota ei
tarvitse editoida — komentorivin `--account` ohittaa sen ja asettaa myös
`$SLURM_JOB_ACCOUNT`-polun oikein):

```sh
cd $TYOTILA
sbatch --account=$PROJEKTI csc/kuuntelija-yksi.sh    # yksi GPU, koko kansio
sbatch --account=$PROJEKTI csc/kuuntelija-array.sh   # 8 GPU:ta, kukin oman siivunsa
```

Seuraa: `squeue --me`. Tuloste menee `kuuntelija_<jobid>.out`-tiedostoon.
Molemmat voi lähettää uudelleen jos aika loppuu kesken — valmiit biisit
ohitetaan. Ajon jälkeen `seff <jobid>` näyttää todellisen GPU-/muistin
käytön.

## Suomennos jälkikäteen

Puhtilla ajetaan vain englanninkieliset kuvaukset (Ollamaa ei ole).
Kun raportit on haettu takaisin omalle koneelle:

```sh
rclone copy $TYOTILA/musiikki s3allas-project_XXXXXXX:BUCKETTI --include "*.txt" -P
# kotona, kun raportit on haettu:
python kuuntelija30b.py ~/Musiikki --vain-suomi --malli gemma3:4b
```

`--vain-suomi` ei analysoi mitään uutta — se lisää KUVAUS SUOMEKSI -osion
niihin raportteihin, joista se puuttuu.
