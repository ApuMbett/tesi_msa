import os
from pydub import AudioSegment

# I risultati del tuo modello
cue_data = {
    "blablabla.mp3": [0.11609977324263039, 56.122630385487525, 106.60281179138322],
    "shotmedown.mp3": [42.63183673469388, 65.0855328798186, 80.10884353741497, 121.41714285714286, 136.3243537414966, 143.8940589569161, 183.01968253968255]
}

cartella_input = "tracks"
cartella_output = "spezzoni_cue"

# Crea la cartella di output se non esiste
os.makedirs(cartella_output, exist_ok=True)

for traccia, cue_points in cue_data.items():
    percorso_file = os.path.join(cartella_input, traccia)
    
    if not os.path.exists(percorso_file):
        print(f"❌ File non trovato: {percorso_file}")
        continue

    print(f"🎧 Caricamento di {traccia}...")
    audio = AudioSegment.from_mp3(percorso_file)

    # Ciclo per creare un segmento tra ogni cue point e il successivo
    for i in range(len(cue_points) - 1):
        start_sec = cue_points[i]
        end_sec = cue_points[i+1]

        # Pydub lavora in millisecondi, quindi moltiplichiamo per 1000
        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)

        # Il taglio magico!
        segmento = audio[start_ms:end_ms]

        # Creiamo un nome file pulito (es: blablabla_0s_a_56s.mp3)
        nome_base = traccia.replace(".mp3", "")
        nome_file_out = f"{nome_base}_{int(start_sec)}s_a_{int(end_sec)}s.mp3"
        percorso_out = os.path.join(cartella_output, nome_file_out)

        # Esportiamo lo spezzone
        segmento.export(percorso_out, format="mp3")
        print(f"  ✅ Salvato: {nome_file_out}")

print("\n🎉 Finito! Trovi tutti i ritagli nella cartella 'spezzoni_cue'")
