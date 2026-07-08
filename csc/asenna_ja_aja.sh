#!/bin/bash
# Kuuntelija Puhti -asennus JA ajo yhdellä komennolla (kirjautumisnoodilla).
#
# Käyttö:   bash csc/asenna_ja_aja.sh <projekti> [bucket]
# Esim:     bash csc/asenna_ja_aja.sh project_2016173 mp3
#
# Idempotentti: jo tehdyt vaiheet ohitetaan, joten voi ajaa uudelleen
# turvallisesti (esim. jos jokin lataus katkesi). Lokittaa joka vaiheen.
# Aja KIRJAUTUMISNOODILLA — laskentanoodilla ei ole internetiä.

set -o pipefail

PROJEKTI="${1:?Anna projektinumero. Käyttö: bash csc/asenna_ja_aja.sh <projekti> [bucket]}"
BUCKET="${2:-mp3}"
TYOTILA=/scratch/$PROJEKTI/kuuntelija
ARKKIT=70   # Puhti V100 = 70, Mahti A100 = 80

# ---- lokitusapurit -----------------------------------------------------
log()  { printf '\n\033[1;36m[%(%H:%M:%S)T] ── %s\033[0m\n' -1 "$*"; }
ok()   { printf '\033[1;32m      ✓ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m      … %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m      ✗ VIRHE: %s\033[0m\n' "$*" >&2; exit 1; }

# ---- module-komento toimimaan myös ei-interaktiivisessa bashissa -------
if ! command -v module &>/dev/null; then
    [ -n "${MODULESHOME:-}" ] && source "$MODULESHOME/init/bash" 2>/dev/null
    command -v module &>/dev/null || die "module-komento ei käytettävissä — aja kirjautumisnoodilla"
fi

log "KUUNTELIJA-ASENNUS  projekti=$PROJEKTI  bucket=$BUCKET"
echo "      kone:    $(hostname)"
echo "      työtila: $TYOTILA"
mkdir -p "$TYOTILA" && cd "$TYOTILA" || die "työtilaan ei pääse"

# ======================================================================
log "1/6  Mallit (Qwen3-Omni-30B-A3B GGUF)"
MALLI=mallit/Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf
MMPROJ=mallit/mmproj-Qwen3-Omni-30B-A3B-Instruct-bf16.gguf
mkdir -p mallit
hae_hf() {  # $1 = tiedostonimi
    HF_XET_HIGH_PERFORMANCE=1 python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF', filename='$1', local_dir='mallit')"
}
if [ -f "$MALLI" ] && [ "$(stat -c%s "$MALLI")" -gt 18000000000 ]; then
    ok "päämalli jo ladattu ($(du -h "$MALLI" | cut -f1))"
else
    warn "ladataan päämalli ~17 Gt (HF Xet -siirto, nopea)…"
    module purge; module load python-data || die "python-data-moduuli puuttuu"
    pip install --user --quiet huggingface_hub hf_transfer hf_xet || die "huggingface_hub-asennus epäonnistui"
    hae_hf "Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf" || die "päämallin lataus epäonnistui"
    ok "päämalli ladattu ($(du -h "$MALLI" | cut -f1))"
fi
if [ -f "$MMPROJ" ] && [ "$(stat -c%s "$MMPROJ")" -gt 2000000000 ]; then
    ok "mmproj (audioenkooderi) jo ladattu"
else
    warn "ladataan mmproj ~2 Gt…"
    module purge; module load python-data 2>/dev/null
    pip install --user --quiet huggingface_hub hf_transfer hf_xet 2>/dev/null
    hae_hf "mmproj-Qwen3-Omni-30B-A3B-Instruct-bf16.gguf" || die "mmproj lataus epäonnistui"
    ok "mmproj ladattu"
fi
# huggingface_hub 1.x jää ~/.local:iin ja rikkoo pytorch-moduulin
# transformersin (vaatii <1.0). Poistetaan — mallit on jo haettu.
if ls "$HOME"/.local/lib/python*/site-packages/huggingface_hub/__init__.py &>/dev/null; then
    warn "poistetaan ~/.local huggingface_hub (ristiriita transformersin kanssa)"
    rm -rf "$HOME"/.local/lib/python*/site-packages/huggingface_hub \
           "$HOME"/.local/lib/python*/site-packages/huggingface_hub-*.dist-info
    ok "poistettu"
fi

# ======================================================================
log "2/6  llama.cpp (CUDA, llama-mtmd-cli)"
CLI=llama.cpp/build/bin/llama-mtmd-cli
if [ -x "$CLI" ]; then
    ok "llama-mtmd-cli jo käännetty"
else
    warn "käännetään llama.cpp CUDA-tuella (arkkitehtuuri $ARKKIT)…"
    module purge; module load gcc cuda cmake || die "käännösmoduulit puuttuvat"
    [ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp || die "llama.cpp kloonaus epäonnistui"
    # Kirjautumisnoodilla ei ole GPU-ajuria; annetaan ajurin stub versioidulla
    # nimellä linkkeriä varten (ajossa GPU-noodin oikea ajuri löytyy).
    CUDA_ROOT=$(dirname "$(dirname "$(command -v nvcc)")")
    mkdir -p cudastub
    ln -sf "$CUDA_ROOT/lib64/stubs/libcuda.so" cudastub/libcuda.so.1
    cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$ARKKIT \
          -DCMAKE_EXE_LINKER_FLAGS="-L$TYOTILA/cudastub -Wl,-rpath-link,$TYOTILA/cudastub" \
          -DCMAKE_SHARED_LINKER_FLAGS="-L$TYOTILA/cudastub -Wl,-rpath-link,$TYOTILA/cudastub" \
          || die "cmake-konfigurointi epäonnistui"
    cmake --build llama.cpp/build --config Release -j 8 --target llama-mtmd-cli || die "käännös epäonnistui"
    [ -x "$CLI" ] && ok "käännetty" || die "binääriä ei syntynyt"
fi

# ======================================================================
log "3/6  Python-ympäristö (venv pytorch-moduulin päälle)"
module purge; module load pytorch || die "pytorch-moduuli puuttuu"
if [ -x .venv/bin/python ]; then
    ok "venv on jo olemassa"
else
    warn "luodaan venv ja asennetaan librosa + soundfile…"
    python3 -m venv --system-site-packages .venv || die "venv-luonti epäonnistui"
    .venv/bin/pip install --quiet librosa soundfile || die "librosa/soundfile-asennus epäonnistui"
    ok "venv luotu"
fi
if .venv/bin/python -c "import torch,transformers,librosa,soundfile,numpy" 2>/dev/null; then
    VERS=$(.venv/bin/python -c "import torch,transformers,librosa;print(torch.__version__,transformers.__version__,librosa.__version__)")
    ok "importit kunnossa: torch/transformers/librosa = $VERS"
else
    die "importit eivät toimi — todennäköisesti ~/.local huggingface_hub -ristiriita (aja skripti uudelleen)"
fi

# ======================================================================
log "4/6  Luokittelijoiden esilataus välimuistiin (laskentanoodi ei lataa)"
export HF_HOME=$TYOTILA/hf-cache
if [ -d "$HF_HOME/hub/models--dima806--music_genres_classification" ] && \
   [ -d "$HF_HOME/hub/models--MIT--ast-finetuned-audioset-10-10-0.4593" ]; then
    ok "luokittelijat jo välimuistissa"
else
    warn "ladataan genre- ja tagimalli ~700 Mt…"
    export HF_HUB_ENABLE_HF_TRANSFER=1
    .venv/bin/python -c "from transformers import pipeline; pipeline('audio-classification', model='dima806/music_genres_classification', device=-1); pipeline('audio-classification', model='MIT/ast-finetuned-audioset-10-10-0.4593', device=-1)" \
        || die "luokittelijoiden lataus epäonnistui"
    ok "luokittelijat välimuistissa"
fi

# ======================================================================
log "5/6  Musiikki Allaksesta ($BUCKET → musiikki/)"
mkdir -p musiikki
command -v rclone &>/dev/null || module load allas 2>/dev/null
if command -v rclone &>/dev/null && rclone lsd "s3allas-$PROJEKTI:" &>/dev/null; then
    rclone copy "s3allas-$PROJEKTI:$BUCKET" musiikki -P || warn "rclone-kopiointi ei täysin onnistunut"
else
    warn "rclone-remotea 's3allas-$PROJEKTI' ei löydy — konfiguroi se MyCSC:n"
    warn "Cloud storage -paneelista, tai kopioi biisit käsin kansioon musiikki/"
fi
N=$(find musiikki -maxdepth 1 -type f \( -iname '*.mp3' -o -iname '*.wav' -o -iname '*.flac' \
     -o -iname '*.m4a' -o -iname '*.ogg' -o -iname '*.opus' \) 2>/dev/null | wc -l)
[ "$N" -gt 0 ] && ok "musiikkikansiossa $N audiotiedostoa" \
               || die "musiikkikansiossa ei audiota — ei ole mitä ajaa"

# ======================================================================
log "6/6  Lähetetään eräajo jonoon"
JOBID=$(sbatch --parsable --account="$PROJEKTI" csc/kuuntelija-yksi.sh) \
    || die "sbatch epäonnistui"
ok "jobi jonossa: $JOBID"

# ----------------------------------------------------------------------
log "VALMIS — asennus tehty, ajo jonossa."
cat <<OHJE

  Seuraa ajoa:
      squeue --me
      tail -f $TYOTILA/kuuntelija_$JOBID.out

  ($N biisiä, arvio ~1-2 min/biisi GPU:lla + jonotusaika.)

  Kun jobi on valmis, hae raportit Allakseen ja suomenna kotona:
      rclone copy $TYOTILA/musiikki s3allas-$PROJEKTI:$BUCKET --include '*.txt' -P
      # kotikoneella:
      python kuuntelija30b.py <kansio> --vain-suomi --malli gemma3:4b

OHJE
