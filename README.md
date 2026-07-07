# Kuuntelija

Skripti, joka kuuntelee kansion audiotiedostot paikallisilla malleilla ja
kirjoittaa jokaisesta kuvauksen samannimiseen `.txt`-tiedostoon
(esim. `biisi.mp3` → `biisi.txt`). Mikään ei lähde verkkoon — kaikki
analyysi tapahtuu omalla koneella. Kehitetty ja testattu Intel-iMacilla
(i5, 4 ydintä, 24 Gt RAM), eli pyörii pelkällä prosessorilla.

## Miten se toimii

1. **GTZAN-genremalli** ja **AudioSet-tagimalli** (Hugging Face)
   tunnistavat genren ja tagit (soittimet, laulu, tunnelma)
2. **Qwen2.5-Omni-7B** (llama.cpp) kuuntelee 30 s pätkän biisin keskeltä
   ja kirjoittaa yksityiskohtaisen englanninkielisen kuvauksen: genre,
   soittimet ja soittotapa, komppi, basso, laulutyyli, sovitus —
   musiikkituotantokehotteen tyyliin. Genremallin arvio annetaan sille
   vihjeeksi, koska genrejen nimeäminen on audio-LLM:n heikkous
   (soittimet ja tunnelman se kuulee itse)
3. **librosa** mittaa tempon ja sävellajin, jotka liitetään kuvauksen
   perään — audio-LLM arvaa ne huonosti, joten ne mitataan erikseen
4. Valinnaisesti (`--suomi`) **Ollama + gemma3:4b** kirjoittaa
   havainnoista myös suomenkielisen kuvauksen

Esimerkki tuotoksesta:

> A fast-paced, lively track with a strong rock influence, featuring a
> prominent electric guitar playing a catchy riff, supported by a steady
> drum pattern and a driving bass line. The vocals are male, with a wide
> range and a passionate, energetic delivery style. The overall
> arrangement is dynamic and engaging. Tempo is 134 BPM in the key of
> C# minor.

## Tarvittavat mallit

### Qwen2.5-Omni-7B (kuvailun päämoottori)

Ladataan käsin `mallit/`-kansioon reposta
[ggml-org/Qwen2.5-Omni-7B-GGUF](https://huggingface.co/ggml-org/Qwen2.5-Omni-7B-GGUF):

| Tiedosto | Koko | Suora latauslinkki |
|---|---|---|
| `Qwen2.5-Omni-7B-Q4_K_M.gguf` | 4,7 Gt | [lataa](https://huggingface.co/ggml-org/Qwen2.5-Omni-7B-GGUF/resolve/main/Qwen2.5-Omni-7B-Q4_K_M.gguf) |
| `mmproj-Qwen2.5-Omni-7B-f16.gguf` | 2,6 Gt | [lataa](https://huggingface.co/ggml-org/Qwen2.5-Omni-7B-GGUF/resolve/main/mmproj-Qwen2.5-Omni-7B-f16.gguf) |

```sh
mkdir -p mallit && cd mallit
curl -LO "https://huggingface.co/ggml-org/Qwen2.5-Omni-7B-GGUF/resolve/main/Qwen2.5-Omni-7B-Q4_K_M.gguf"
curl -LO "https://huggingface.co/ggml-org/Qwen2.5-Omni-7B-GGUF/resolve/main/mmproj-Qwen2.5-Omni-7B-f16.gguf"
```

**Huom:** audioenkooderista (mmproj) tarvitaan nimenomaan **f16-versio**.
Q8-kvantisoitu enkooderi riittänee puheelle, mutta pilaa musiikin
kuuntelun täysin — testissä folk-rock kuvautui "glitchy-elektroniikaksi
ilman laulua". Myös Qwen2-Audio-7B kokeiltiin ja hylättiin selvästi
huonompana musiikin kuvailussa.

### Luokittelijat (latautuvat itsestään)

Nämä Hugging Face -mallit latautuvat ensimmäisellä ajolla automaattisesti
välimuistiin (`~/.cache/huggingface`), yhteensä n. 700 Mt:

- [dima806/music_genres_classification](https://huggingface.co/dima806/music_genres_classification)
  — GTZAN-genretunnistus (10 genreä)
- [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
  — AudioSet-tagit (527 luokkaa: soittimet, laulu, tunnelma)

### Gemma 3 4B (valinnainen, `--suomi`-lippua varten)

```sh
ollama pull gemma3:4b   # n. 3,3 Gt
```

## Asennus

```sh
brew install ffmpeg llama.cpp        # llama.cpp tuo llama-mtmd-cli-komennon
brew install ollama                  # valinnainen, vain --suomi-lippua varten
brew services start ollama

python3 -m venv .venv
.venv/bin/pip install "numpy<2" "torch==2.2.2" "transformers==4.44.2" librosa soundfile
```

torch on pinnattu versioon 2.2.2, koska se on viimeinen Intel-Macia
(x86_64) tukeva versio. Apple Silicon- tai Linux-koneella uudempikin käy.

## Käyttö

```sh
.venv/bin/python kuuntelija.py            # käsittelee tämän kansion
.venv/bin/python kuuntelija.py ~/Musiikki # tai jonkin muun kansion
.venv/bin/python kuuntelija.py --suomi    # lisäksi kuvaus suomeksi
.venv/bin/python kuuntelija.py --force    # kirjoita vanhojen .txt:iden yli
```

Jo kuvaillut biisit ohitetaan automaattisesti, joten skriptin voi ajaa
samaan kansioon uudelleen kun sinne ilmestyy uutta musiikkia.
Intel-iMacilla Qwenin kuuntelu kestää n. 4–5 min/biisi, joten isot
kansiot kannattaa ajaa yön yli. Jos Qwen-mallit puuttuvat, skripti
kirjoittaa silti analyysitiedot (genre, tempo, tagit) mutta jättää
kuvailun pois.
