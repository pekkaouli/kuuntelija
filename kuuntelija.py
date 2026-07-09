#!/usr/bin/env python3
"""
kuuntelija.py — kuuntelee kansion audiotiedostot paikallisilla malleilla ja
kirjoittaa jokaisesta kuvauksen samannimiseen .txt-tiedostoon.

Vaiheet per biisi:
  1. ffmpeg poimii biisistä näytepätkät
  2. Qwen2.5-Omni (llama.cpp) kuuntelee 30 s pätkän ja kirjoittaa
     yksityiskohtaisen kuvauksen (genre, soittimet, komppi, laulu, sovitus)
  3. GTZAN-genremalli ja AudioSet-tagimalli antavat vertailuarvion
  4. librosa mittaa tempon, sävellajin ja energian — nämä liitetään
     kuvauksen perään, koska audio-LLM arvaa ne huonosti
  5. valinnaisesti (--suomi) Ollama-kielimalli kirjoittaa kuvauksen suomeksi

Käyttö:
  .venv/bin/python kuuntelija.py [kansio] [--suomi] [--force]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

AUDIO_PAATTEET = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aiff", ".opus"}
GENRE_MALLI = "dima806/music_genres_classification"
TAGI_MALLI = "MIT/ast-finetuned-audioset-10-10-0.4593"
OLLAMA_URL = "http://localhost:11434"
NAYTE_SEKUNNIT = 12
SR = 16000

SKRIPTIKANSIO = Path(__file__).resolve().parent
MALLIKANSIO = SKRIPTIKANSIO / "mallit"
QWEN_MALLI = MALLIKANSIO / "Qwen2.5-Omni-7B-Q4_K_M.gguf"
# mmproj f16, ei Q8: kvantisoitu audioenkooderi pilaa musiikin kuuntelun
QWEN_MMPROJ = MALLIKANSIO / "mmproj-Qwen2.5-Omni-7B-f16.gguf"
QWEN_NAYTE_SEKUNNIT = 30  # Qwenin audioenkooderi kuulee enintään 30 s


def _etsi_llama_cli():
    """Paikanna llama-mtmd-cli ilman PATH-riippuvuutta: ensin
    KUUNTELIJA_LLAMA_CLI-ympäristömuuttuja, sitten PATH (esim. Homebrew),
    lopuksi skriptin viereen käännetty llama.cpp."""
    ehdokas = os.environ.get("KUUNTELIJA_LLAMA_CLI")
    if ehdokas and Path(ehdokas).is_file():
        return ehdokas
    polulta = shutil.which("llama-mtmd-cli")
    if polulta:
        return polulta
    paikallinen = SKRIPTIKANSIO / "llama.cpp" / "build" / "bin" / "llama-mtmd-cli"
    if paikallinen.is_file() and os.access(paikallinen, os.X_OK):
        return str(paikallinen)
    return None


QWEN_CLI = _etsi_llama_cli()

# Genren nimeäminen on Qwenin heikkous, joten luokittelijoiden arvio
# annetaan vihjeeksi; soittimet, kompin ja laulun se kuulee itse.
QWEN_KEHOTE = (
    "You are listening to an excerpt of a song. Describe it in detail for "
    "a music production prompt. Cover: genre and subgenre influences; the "
    "instruments and what they play (riffs, chords, patterns) and their "
    "tone; the drum pattern and cymbal work; the bass line; the vocals "
    "(gender, range, delivery style); the overall arrangement. Format: one "
    "paragraph of short comma-separated clauses, no full sentences, like a "
    "prompt for a music generation model. Do not guess tempo or key. "
    "A rough genre classifier suggested: {vihjeet} — it may be wrong, "
    "trust what you hear."
)

# AudioSetin ylägeneeriset luokat, jotka eivät kerro biisistä mitään
TYLSAT_TAGIT = {"Music", "Speech", "Musical instrument", "Sound effect", "Inside, small room"}

GENRE_SUOMEKSI = {
    "blues": "blues", "classical": "klassinen", "country": "country",
    "disco": "disco", "hiphop": "hip hop", "jazz": "jazz", "metal": "metalli",
    "pop": "pop", "reggae": "reggae", "rock": "rock",
}

SAVELET = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Krumhansl-Schmucklerin sävellajiprofiilit
DUURI_PROFIILI = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MOLLI_PROFIILI = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def aja(komento):
    # encoding pakotettu: Windowsissa oletus olisi cp1252, joka rikkoo UTF-8:n
    return subprocess.run(komento, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", check=True).stdout


def kesto_sekunteina(tiedosto):
    out = aja(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "csv=p=0", str(tiedosto)])
    return float(out.strip())


def metatiedot(tiedosto):
    """Lukee ID3-tagit (artisti, nimi, albumi) ffproben kautta."""
    out = aja(["ffprobe", "-v", "error", "-show_entries", "format_tags",
               "-of", "json", str(tiedosto)])
    tagit = json.loads(out).get("format", {}).get("tags", {})
    tagit = {k.lower(): v for k, v in tagit.items()}
    return {avain: tagit.get(avain) for avain in ("artist", "title", "album", "date", "genre")}


def poimi_naytteet(tiedosto, kesto):
    """Purkaa ffmpegillä 16 kHz mononäytteet biisin eri kohdista."""
    if kesto <= NAYTE_SEKUNNIT * 3:
        kohdat = [0.0]
    else:
        kohdat = [kesto * 0.10, kesto * 0.40, kesto * 0.70]
    naytteet = []
    for kohta in kohdat:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            polku = tmp.name
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{kohta:.2f}",
                        "-t", str(NAYTE_SEKUNNIT), "-i", str(tiedosto),
                        "-ac", "1", "-ar", str(SR), polku], check=True)
        import soundfile as sf
        data, _ = sf.read(polku, dtype="float32")
        Path(polku).unlink()
        if len(data) > SR:  # alle sekunnin pätkät pois
            naytteet.append(data)
    return naytteet


def lataa_luokittelijat():
    from transformers import pipeline
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    print("  Ladataan mallit (ensimmäisellä kerralla latautuu verkosta)...")
    genre = pipeline("audio-classification", model=GENRE_MALLI, device=-1)
    tagit = pipeline("audio-classification", model=TAGI_MALLI, device=-1)
    return genre, tagit


def luokittele(luokittelija, naytteet, top_k):
    """Ajaa luokittelijan kaikille näytteille ja keskiarvottaa pisteet."""
    pisteet = {}
    for nayte in naytteet:
        for tulos in luokittelija({"raw": nayte, "sampling_rate": SR}, top_k=top_k):
            pisteet.setdefault(tulos["label"], []).append(tulos["score"])
    keskiarvot = {nimi: sum(p) / len(naytteet) for nimi, p in pisteet.items()}
    return sorted(keskiarvot.items(), key=lambda x: -x[1])


def analysoi_piirteet(naytteet):
    """Tempo, sävellaji ja energia librosalla."""
    import librosa
    import librosa.feature.rhythm
    audio = np.concatenate(naytteet)
    tempo = float(np.atleast_1d(librosa.feature.rhythm.tempo(y=audio, sr=SR))[0])

    kroma = librosa.feature.chroma_cqt(y=audio, sr=SR).mean(axis=1)
    parhaat = []
    for siirto in range(12):
        pyoritetty = np.roll(kroma, -siirto)
        parhaat.append((np.corrcoef(pyoritetty, DUURI_PROFIILI)[0, 1], siirto, "duuri"))
        parhaat.append((np.corrcoef(pyoritetty, MOLLI_PROFIILI)[0, 1], siirto, "molli"))
    _, savel, laji = max(parhaat)
    savellaji = f"{SAVELET[savel]}-{laji}"

    rms = float(librosa.feature.rms(y=audio).mean())
    kirkkaus = float(librosa.feature.spectral_centroid(y=audio, sr=SR).mean())
    savellaji_en = f"{SAVELET[savel]} {'Major' if laji == 'duuri' else 'minor'}"
    return {"tempo": round(tempo), "savellaji": savellaji,
            "savellaji_en": savellaji_en, "energia": rms, "kirkkaus": kirkkaus}


def energia_sanoiksi(piirteet):
    kuvaus = []
    kuvaus.append("energinen ja tiivis soundi" if piirteet["energia"] > 0.15
                  else "rauhallinen ja ilmava soundi" if piirteet["energia"] < 0.05
                  else "keskivahva soundi")
    kuvaus.append("kirkas ja diskanttivoittoinen" if piirteet["kirkkaus"] > 3000
                  else "tumma ja bassovoittoinen" if piirteet["kirkkaus"] < 1200
                  else "tasapainoinen taajuusjakauma")
    return ", ".join(kuvaus)


def qwen_kaytettavissa():
    return QWEN_MALLI.exists() and QWEN_MMPROJ.exists() and QWEN_CLI is not None


def kuuntele_qwenilla(tiedosto, kesto, vihjeet):
    """Qwen2.5-Omni kuuntelee 30 s biisin keskeltä ja kuvailee kuulemansa."""
    kohta = max(0.0, kesto * 0.4 - QWEN_NAYTE_SEKUNNIT / 2)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        klippi = tmp.name
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{kohta:.2f}",
                        "-t", str(QWEN_NAYTE_SEKUNNIT), "-i", str(tiedosto),
                        "-ac", "1", "-ar", str(SR), klippi], check=True)
        tulos = subprocess.run(
            [QWEN_CLI, "-m", str(QWEN_MALLI), "--mmproj", str(QWEN_MMPROJ),
             "--audio", klippi, "-p", QWEN_KEHOTE.format(vihjeet=vihjeet),
             "-n", "300", "--temp", "0.3", "-t", "4"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            check=True, timeout=1800)
        return tulos.stdout.strip()
    finally:
        Path(klippi).unlink(missing_ok=True)


def ollama_kaytettavissa(malli):
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as vastaus:
            mallit = [m["name"] for m in json.load(vastaus).get("models", [])]
        return any(m == malli or m.startswith(malli + ":") for m in mallit)
    except Exception:
        return False


def kirjoita_kuvaus_ollamalla(malli, fakta_arkki):
    kehote = (
        "Olet musiikkitoimittaja. Kirjoita alla olevien analyysitietojen pohjalta "
        "suomeksi kahden kappaleen mittainen, elävä kuvaus tästä kappaleesta. "
        "Kerro genrestä, tunnelmasta, soundista ja soittimista. Älä keksi faktoja, "
        "joita tiedoissa ei ole (esim. artistin taustaa tai sanoituksia), äläkä "
        "luettele lukuja mekaanisesti — kudo tiedot luontevaksi tekstiksi.\n\n"
        f"Analyysitiedot:\n{fakta_arkki}\n\nKuvaus:"
    )
    pyynto = json.dumps({"model": malli, "prompt": kehote, "stream": False,
                         "options": {"temperature": 0.7}}).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=pyynto,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as vastaus:
        return json.load(vastaus)["response"].strip()


def kasittele(tiedosto, genre_malli, tagi_malli, ollama_malli, qwen_ok):
    kesto = kesto_sekunteina(tiedosto)
    meta = metatiedot(tiedosto)
    naytteet = poimi_naytteet(tiedosto, kesto)

    print("  Tunnistetaan genreä ja tageja...")
    genret = luokittele(genre_malli, naytteet, top_k=5)
    tagit = [(nimi, p) for nimi, p in luokittele(tagi_malli, naytteet, top_k=12)
             if nimi not in TYLSAT_TAGIT][:8]
    piirteet = analysoi_piirteet(naytteet)

    kuvailu = None
    if qwen_ok:
        print("  Qwen2.5-Omni kuuntelee (kestää muutaman minuutin)...")
        vihjeet = ", ".join(f"{n} {p:.0%}" for n, p in genret[:3] if p > 0.10)
        kuvailu = kuuntele_qwenilla(tiedosto, kesto, vihjeet)
        kuvailu = (f"{kuvailu.rstrip('.')}. "
                   f"Tempo is {piirteet['tempo']} BPM "
                   f"in the key of {piirteet['savellaji_en']}.")

    paagenre = GENRE_SUOMEKSI.get(genret[0][0], genret[0][0])
    genre_rivi = ", ".join(f"{GENRE_SUOMEKSI.get(n, n)} ({p:.0%})" for n, p in genret[:3])
    tagi_rivi = ", ".join(f"{n} ({p:.0%})" for n, p in tagit)

    fakta_arkki = "\n".join(rivi for rivi in [
        f"Tiedosto: {tiedosto.name}",
        f"Artisti: {meta['artist']}" if meta.get("artist") else None,
        f"Kappale: {meta['title']}" if meta.get("title") else None,
        f"Albumi: {meta['album']}" if meta.get("album") else None,
        f"Kesto: {int(kesto // 60)} min {int(kesto % 60)} s",
        f"Genre (luokittelijan arvio): {genre_rivi}",
        f"Äänitunnisteet: {tagi_rivi}",
        f"Tempo: noin {piirteet['tempo']} bpm",
        f"Sävellaji (arvio): {piirteet['savellaji']}",
        f"Soundi: {energia_sanoiksi(piirteet)}",
    ] if rivi)

    suomennos = None
    if ollama_malli:
        print(f"  Kielimalli ({ollama_malli}) kirjoittaa kuvausta suomeksi...")
        aineisto = fakta_arkki
        if kuvailu:
            aineisto += f"\nKuuntelijan havainnot (englanniksi): {kuvailu}"
        suomennos = kirjoita_kuvaus_ollamalla(ollama_malli, aineisto)

    osat = [f"KUUNTELIJAN RAPORTTI\n{'=' * 60}\n{fakta_arkki}"]
    if kuvailu:
        osat.append(f"KUVAILU\n{'-' * 60}\n{kuvailu}")
    if suomennos:
        osat.append(f"KUVAUS SUOMEKSI\n{'-' * 60}\n{suomennos}")
    if not kuvailu and not suomennos:
        osat.append(f"KUVAUS\n{'-' * 60}\n(Qwen-malli ja Ollama eivät olleet "
                    "käytettävissä — yllä pelkät analyysitiedot.)")

    ulos = tiedosto.with_suffix(".txt")
    ulos.write_text("\n\n".join(osat) + "\n", encoding="utf-8")
    print(f"  Valmis: {ulos.name}  (genre: {paagenre})")


def main():
    parser = argparse.ArgumentParser(description="Kuuntelee kansion biisit ja kuvailee ne.")
    parser.add_argument("kansio", nargs="?", default=".", help="kansio jossa audiotiedostot")
    parser.add_argument("--suomi", action="store_true",
                        help="kirjoita lisäksi suomenkielinen kuvaus Ollamalla")
    parser.add_argument("--malli", default="gemma3:4b", help="Ollama-malli --suomi-kuvausta varten")
    parser.add_argument("--force", action="store_true", help="kirjoita olemassa olevien .txt:iden yli")
    args = parser.parse_args()

    kansio = Path(args.kansio)
    tiedostot = sorted(t for t in kansio.iterdir()
                       if t.suffix.lower() in AUDIO_PAATTEET)
    if not tiedostot:
        print(f"Ei audiotiedostoja kansiossa {kansio.resolve()}")
        return

    if not args.force:
        ohitetut = [t for t in tiedostot if t.with_suffix(".txt").exists()]
        tiedostot = [t for t in tiedostot if not t.with_suffix(".txt").exists()]
        for t in ohitetut:
            print(f"Ohitetaan {t.name} (kuvaus on jo olemassa)")
    if not tiedostot:
        return

    qwen_ok = qwen_kaytettavissa()
    if not qwen_ok:
        syy = ("mallitiedostot puuttuvat kansiosta " + str(MALLIKANSIO)
               if not (QWEN_MALLI.exists() and QWEN_MMPROJ.exists())
               else "llama-mtmd-cli ei ratkennut (aseta KUUNTELIJA_LLAMA_CLI "
                    "tai lisää binääri PATHiin)")
        print(f"Huom: Qwen-kuvailu jää pois — {syy}.")

    ollama_malli = None
    if args.suomi:
        ollama_malli = args.malli if ollama_kaytettavissa(args.malli) else None
        if not ollama_malli:
            print(f"Huom: Ollama tai malli '{args.malli}' ei vastaa osoitteessa "
                  f"{OLLAMA_URL} — suomenkielinen kuvaus jää pois.")

    genre_malli, tagi_malli = lataa_luokittelijat()
    for i, tiedosto in enumerate(tiedostot, 1):
        print(f"[{i}/{len(tiedostot)}] {tiedosto.name}")
        try:
            kasittele(tiedosto, genre_malli, tagi_malli, ollama_malli, qwen_ok)
        except Exception as virhe:
            print(f"  VIRHE: {virhe}", file=sys.stderr)


if __name__ == "__main__":
    main()
