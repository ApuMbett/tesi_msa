import os

# I risultati del tuo modello
# cue_data = {
#     "blablabla.mp3": [0.11609977324263039, 56.122630385487525, 106.60281179138322],
#     "shotmedown.mp3": [42.63183673469388, 65.0855328798186, 80.10884353741497, 121.41714285714286, 136.3243537414966, 143.8940589569161, 183.01968253968255]
# }

def read_cue_points_from_json(json_path):
    import json
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

cartella_output = "etichette_audacity"
os.makedirs(cartella_output, exist_ok=True)
cue_data = read_cue_points_from_json("./tracks/_cue_points.json")
for traccia, cue_points in cue_data.items():
    nome_txt = traccia.replace(".mp3", "_labels.txt")
    percorso_out = os.path.join(cartella_output, nome_txt)

    with open(percorso_out, "w") as f:
        for i, cp in enumerate(cue_points):
            # Audacity vuole: tempo_inizio (tab) tempo_fine (tab) Nome
            # Mettiamo inizio e fine uguali così crea un singolo "punto" e non una regione
            time = cp["time"]
            score = cp["score"]
            f.write(f"{time:.6f}\t{time:.6f}\tCue {i+1} (score: {score:.4f})\n")

    print(f"✅ Creato file etichette: {nome_txt}")
