# Kuuntelija CSC:n Puhtilla (eräajo)

Ohjeet `kuuntelija30b.py`:n ajamiseen Slurm-eräajona Puhtin GPU-noodeilla.
Puhtin V100:ssa on 32 Gt muistia, joten koko Q4-malli mahtuu näytönohjaimeen
(`KUUNTELIJA_CPU_MOE=0`) — biisin analyysi on selvästi nopeampaa kuin
kotikoneella. Sama toimii Mahtilla (A100 40 Gt) partitiota vaihtamalla.

> **Huom:** skriptit on kirjoitettu Windows-koneella eikä niitä ole vielä
> ajettu Puhtilla — ensimmäisellä kerralla voi tulla pientä säätöä esim.
> moduulien nimiin. Kustannus on luokkaa 60 BU / GPU-tunti eli ~400 BU
> tuhannen biisin kansiolle.

## Kertavalmistelut (kirjautumisnoodilla)

Kaikki lataukset tehdään kirjautumisnoodilla — **laskentanoodeilla ei ole
internet-yhteyttä**.

```sh
# 0. Muuttujat (vaihda oma projektinumerosi)
export PROJEKTI=project_XXXXXXX
export TYOTILA=/scratch/$PROJEKTI/kuuntelija

# 1. Repo ja mallit
mkdir -p $TYOTILA && cd $TYOTILA
git clone https://github.com/pekkaouli/kuuntelija.git .
mkdir -p mallit && cd mallit
curl -LO "https://huggingface.co/ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF/resolve/main/Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf"
curl -LO "https://huggingface.co/ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF/resolve/main/mmproj-Qwen3-Omni-30B-A3B-Instruct-bf16.gguf"
cd $TYOTILA

# 2. llama.cpp CUDA-tuella (arkkitehtuurit: 70 = Puhti V100, 80 = Mahti A100)
module load gcc cuda cmake git
git clone https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="70;80"
cmake --build llama.cpp/build --config Release -j 8 --target llama-mtmd-cli

# 3. Python-ympäristö CSC:n pytorch-moduulin päälle
module load pytorch
python3 -m venv --system-site-packages .venv
.venv/bin/pip install transformers librosa soundfile

# 4. ffmpeg (staattinen binääri, jos moduulia ei ole: module spider ffmpeg)
mkdir -p bin && cd bin
curl -L "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz" | tar -xJ --strip-components=1
cd $TYOTILA

# 5. Esilataa HF-luokittelijat välimuistiin (laskentanoodi ei voi ladata!)
export HF_HOME=$TYOTILA/hf-cache
.venv/bin/python - <<'EOF'
from transformers import pipeline
pipeline("audio-classification", model="dima806/music_genres_classification", device=-1)
pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593", device=-1)
print("Luokittelijat välimuistissa.")
EOF

# 6. Siirrä musiikki työtilaan (omalta koneelta: rsync tai Allas)
mkdir -p musiikki
# rsync -av ~/Musiikki/ puhti.csc.fi:$TYOTILA/musiikki/
```

## Ajo

Muokkaa skriptien alkuun oma projektinumero ja lähetä jonoon:

```sh
sbatch csc/kuuntelija-yksi.sh      # yksi GPU, käsittelee kansion alusta loppuun
sbatch csc/kuuntelija-array.sh     # 8 rinnakkaista GPU:ta, kukin oman siivunsa
```

Molemmat voi lähettää uudelleen jos aika loppuu kesken — valmiit biisit
ohitetaan automaattisesti. Jonotilanne: `squeue --me`.

## Suomennos jälkikäteen

Puhtilla ajetaan vain englanninkieliset kuvaukset (Ollamaa ei ole).
Kun raportit on haettu takaisin omalle koneelle, suomennos täydennetään:

```sh
rsync -av puhti.csc.fi:$TYOTILA/musiikki/ ~/Musiikki/   # .txt:t mukana
python kuuntelija30b.py ~/Musiikki --vain-suomi --malli gemma3:4b
```

`--vain-suomi` ei analysoi mitään uutta — se vain lisää KUVAUS
SUOMEKSI -osion niihin raportteihin, joista se puuttuu.
