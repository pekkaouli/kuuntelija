# Kuuntelija

Skripti, joka kuuntelee kansion audiotiedostot paikallisilla malleilla ja
kirjoittaa jokaisesta kuvauksen samannimiseen `.txt`-tiedostoon
(esim. `biisi.mp3` → `biisi.txt`). Mikään ei lähde verkkoon — kaikki
analyysi tapahtuu omalla koneella.

Repossa on **kaksi linjaa** saman putken ympärillä:

| | `kuuntelija.py` (kevyt) | `kuuntelija30b.py` (tehokas) |
|---|---|---|
| Kuvailumalli | Qwen2.5-Omni-7B (Q4) | Qwen3-Omni-30B-A3B-Instruct (Q4, MoE) |
| Kuulee biisistä | 30 s keskeltä | koko biisin (max 5 min) |
| Kuvauksen tyyli | tagimainen tuotantoprompti | proosaa, myös rakenteen kaari (intro→säkeistö→kertosäe→silta→outro) |
| Muistivaatimus | ~8 Gt RAM | ~32 Gt RAM (GPU valinnainen, nopeuttaa ~2×) |
| Mitattu nopeus | ~4–5 min/biisi (Intel i5, 4 ydintä) | ~45 s/biisi (RTX 5070 Ti 12 Gt + Core Ultra 9), ~97 s pelkällä CPU:lla |
| Kehitetty | Intel-iMac (24 Gt RAM) | Windows 11 -läppäri (31 Gt RAM, 12 Gt VRAM) |

Sama komentorivikäyttö molemmissa. 30B-malli on myös selvästi
luotettavampi: testibiisi, jonka 7B kuvasi virheellisesti elektroniseksi
vocoder-popiksi, tunnistui 30B:llä oikein akustiseksi indie folkiksi —
sama tulkinta kuin isolla pilvimallilla (Gemini), mutta paikallisesti.

## Miten se toimii

1. **GTZAN-genremalli** ja **AudioSet-tagimalli** (Hugging Face)
   tunnistavat genren ja tagit (soittimet, laulu, tunnelma)
2. **Qwen-Omni** (llama.cpp) kuuntelee biisin ja kirjoittaa
   englanninkielisen kuvauksen. Genremallin arvio annetaan sille
   vihjeeksi, koska genrejen nimeäminen on audio-LLM:n heikkous
   (soittimet ja tunnelman se kuulee itse)
3. **librosa** mittaa tempon ja sävellajin, jotka liitetään kuvauksen
   perään — audio-LLM arvaa ne huonosti, joten ne mitataan erikseen
4. Valinnaisesti (`--suomi`) **Ollama-kielimalli** kirjoittaa
   havainnoista myös suomenkielisen kuvauksen

Esimerkki 30B-linjan tuotoksesta (lyhennetty):

> This is a beautifully understated piece of modern folk-pop, where
> intimacy and restraint create a powerful atmosphere. — — Her delivery
> is nuanced, shifting from a soft, almost hesitant tone in the verses
> to a more resonant, emotionally charged performance in the choruses,
> where the layered harmonies add depth. The song builds gradually, with
> the instrumentation swelling slightly in the bridge — — Tempo is
> 117 BPM in the key of C Major.

## Tarvittavat mallit

Ladataan käsin `mallit/`-kansioon. Vain käyttämäsi linjan mallit
tarvitaan.

### Kevyt linja: Qwen2.5-Omni-7B

Repo [ggml-org/Qwen2.5-Omni-7B-GGUF](https://huggingface.co/ggml-org/Qwen2.5-Omni-7B-GGUF):

| Tiedosto | Koko |
|---|---|
| `Qwen2.5-Omni-7B-Q4_K_M.gguf` | 4,7 Gt |
| `mmproj-Qwen2.5-Omni-7B-f16.gguf` | 2,6 Gt |

### Tehokas linja: Qwen3-Omni-30B-A3B-Instruct

Repo [ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF](https://huggingface.co/ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF):

| Tiedosto | Koko |
|---|---|
| `Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf` | 17,3 Gt |
| `mmproj-Qwen3-Omni-30B-A3B-Instruct-bf16.gguf` | 2,1 Gt |

```sh
mkdir -p mallit && cd mallit
curl -LO "https://huggingface.co/ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF/resolve/main/Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf"
curl -LO "https://huggingface.co/ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF/resolve/main/mmproj-Qwen3-Omni-30B-A3B-Instruct-bf16.gguf"
```

**Huom:** audioenkooderista (mmproj) tarvitaan aina **täystarkka versio**
(f16/bf16). Q8-kvantisoitu enkooderi riittänee puheelle, mutta pilaa
musiikin kuuntelun täysin — testissä folk-rock kuvautui
"glitchy-elektroniikaksi ilman laulua". Myös Qwen2-Audio-7B kokeiltiin
ja hylättiin selvästi huonompana musiikin kuvailussa.

### Luokittelijat (latautuvat itsestään)

Nämä Hugging Face -mallit latautuvat ensimmäisellä ajolla automaattisesti
välimuistiin (`~/.cache/huggingface`), yhteensä n. 700 Mt:

- [dima806/music_genres_classification](https://huggingface.co/dima806/music_genres_classification)
  — GTZAN-genretunnistus (10 genreä)
- [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
  — AudioSet-tagit (527 luokkaa: soittimet, laulu, tunnelma)

### Ollama-malli (valinnainen, `--suomi`-lippua varten)

```sh
ollama pull gemma3:4b    # oletus, n. 3,3 Gt
```

Minkä tahansa muun Ollama-mallin voi valita `--malli`-lipulla
(esim. `--malli gemma4:e4b`).

## Asennus

### macOS

```sh
brew install ffmpeg llama.cpp        # llama.cpp tuo llama-mtmd-cli-komennon
brew install ollama                  # valinnainen, vain --suomi-lippua varten
brew services start ollama

python3 -m venv .venv
.venv/bin/pip install "numpy<2" "torch==2.2.2" "transformers==4.44.2" librosa soundfile
```

torch on pinnattu versioon 2.2.2, koska se on viimeinen Intel-Macia
(x86_64) tukeva versio. Apple Silicon -koneella uudempikin käy.

### Windows

```powershell
winget install Gyan.FFmpeg

# llama.cpp: lataa GitHub-releasesta CUDA-paketti (NVIDIA-kortille) tai
# cpu-x64-paketti, pura esim. kansioon C:\Tools\llama.cpp ja lisää PATHiin.
# CUDA-versioon tarvitaan myös samalta release-sivulta cudart-*.zip samaan
# kansioon. https://github.com/ggml-org/llama.cpp/releases

python -m venv .venv
.venv\Scripts\pip install torch transformers librosa soundfile
```

Windowsilla/uudella Pythonilla (3.13+) versiopinnauksia ei tarvita —
tuoreet torch/transformers/librosa toimivat (testattu Python 3.14,
torch 2.12, transformers 5.13).

## Käyttö

```sh
.venv/bin/python kuuntelija30b.py            # käsittelee tämän kansion
.venv/bin/python kuuntelija30b.py ~/Musiikki # tai jonkin muun kansion
.venv/bin/python kuuntelija30b.py --suomi    # lisäksi kuvaus suomeksi
.venv/bin/python kuuntelija30b.py --force    # kirjoita vanhojen .txt:iden yli
```

(Kevyt linja: sama mutta `kuuntelija.py`. Windowsilla
`.venv\Scripts\python`.)

Jo kuvaillut biisit ohitetaan automaattisesti, joten skriptin voi ajaa
samaan kansioon uudelleen kun sinne ilmestyy uutta musiikkia. Jos
Qwen-mallit puuttuvat, skripti kirjoittaa silti analyysitiedot (genre,
tempo, tagit) mutta jättää kuvailun pois.

30B-linjassa on lisäksi eräajoja tukevat liput:

```sh
python kuuntelija30b.py musiikki --siivu 2/8     # käsittele joka 8. tiedosto
                                                 # 2:sta alkaen (rinnakkaisajo)
python kuuntelija30b.py musiikki --vain-suomi    # täydennä suomennokset
                                                 # valmiisiin raportteihin
```

## Eräajo CSC:n Puhtilla

Ison musiikkikansion voi ajaa Slurm-eräajona CSC:n Puhti-superkoneella,
jonka V100-näytönohjaimeen (32 Gt) koko malli mahtuu kerralla —
`KUUNTELIJA_CPU_MOE=0`-ympäristömuuttujalla. Valmiit sbatch-skriptit
(yksittäisjobi ja 8 GPU:n array) ja asennusohjeet ovat
[csc/](csc/)-kansiossa. Suomennos tehdään jälkikäteen omalla koneella
`--vain-suomi`-lipulla, koska laskentanoodeilla ei ajeta Ollamaa.

## Optimointi: 30B-malli 12 Gt näytönohjaimella

Qwen3-Omni-30B on MoE-malli (Mixture of Experts): painoja on 30 Gt:n
edestä, mutta vain ~3 mrd parametria on aktiivisena per token. Siksi se
pyörii käyttökelpoisella nopeudella pelkällä CPU:llakin — ja vielä
nopeammin, kun llama.cpp:n MoE-jako hyödyntää pienenkin näytönohjaimen:
`-ngl 99` vie kaikki kerrokset GPU:lle, mutta `--n-cpu-moe N` jättää
ensimmäisten N kerroksen isot experttipainot RAMiin.

Mitatut ajat (koko 4 min biisi, RTX 5070 Ti Laptop 12 Gt + Core Ultra 9
275HX, Q4_K_M):

| Asetus | Experttejä CPU:lla | Kesto | VRAM-huippu |
|---|---|---|---|
| `-ngl 0` (pelkkä CPU) | 48/48 | 97 s | 0 |
| `-ngl 99 --cpu-moe` | 48/48 | 77 s | ~4 Gt |
| `-ngl 99 --n-cpu-moe 36` | 36/48 | 57 s | ~8 Gt |
| `-ngl 99 --n-cpu-moe 30` | 30/48 | **44 s** | ~11,5 Gt |
| `-ngl 99 --n-cpu-moe 24` | 24/48 | kaatuu | muisti loppuu (12 Gt) |

Skriptin oletus on `--n-cpu-moe 32`: käytännössä sama nopeus kuin 30,
mutta ~1,5 Gt pelivaraa muille ohjelmille. Jos näytönohjaimessa on
enemmän muistia, pienennä `QWEN_CPU_MOE`-vakiota; jos vähemmän tai ei
ollenkaan, aseta `QWEN_GPU_KERROKSET = 0`.

Muut optimointihavainnot:

- **Konteksti**: `-c 8192` riittää 5 minuutin audiolle (enkooderi tuottaa
  ~12,5 tokenia/s audiota) + kehotteelle + pitkälle vastaukselle.
  Skripti katkaisee pidemmät biisit keskeltä 5 minuuttiin.
- **Puhesynteesiosaa ei tarvitse erikseen kytkeä pois**: Qwen3-Omnin
  "talker" (Transformersissa `disable_talker()`, säästäisi ~10 Gt) ei
  sisälly GGUF-muunnokseen lainkaan — llama.cpp ajaa vain tekstiä
  tuottavan thinker-osan. Siksi 30B-malli mahtuu 17 Gt:iin.
- **Kuvauskieli**: malli kuulee yhtä hyvin molemmilla kielillä mutta
  kirjoittaa englantia selvästi sujuvammin kuin suomea. Siksi Qwen
  kirjoittaa aina englanniksi ja suomennos tehdään erillisellä
  kielimallilla (`--suomi`).
- **Windows-merkistöt**: aliprosessien tuloste luetaan pakotetusti
  UTF-8:na — muuten Windowsin cp1252-oletus rikkoo ajatusviivat ja
  ääkköset.
