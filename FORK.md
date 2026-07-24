# Passer du repo ostris officiel à ce fork

Ce dépôt est un fork de [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit)
avec des ajouts orientés **krea2 sur GPU 12 Go (RTX 4070 Ti)** :

- **LoRAs de sampling** : empiler des loras (turbo + style) sur les samples, global ou par sample.
- **Auto layer-offload** : `layer_offloading_transformer_percent: auto` (plus de réglage à tâtons), + recovery auto sur OOM.
- **Offload déterministe** : les plus grosses couches d'abord (reproductible, plus de OOM par tirage malchanceux).
- **TF32** activé au démarrage (opt-out `AITK_TF32=0`).
- **krea2 offline / local** : text encoder / VAE / checkpoint depuis le cache local (`model_kwargs.offline: true`), modèle raw custom local.
- **Rapport VRAM** + `vram_profile.json`, **`paged_adamw8bit`**, **`torch.compile`** optionnel, **guide quant** (convrot8…).
- **UI** : champs Offline / Text Encoder Path / VAE Path pour krea2.
- Exemple prêt : [`config/examples/train_lora_krea2_12gb.yaml`](config/examples/train_lora_krea2_12gb.yaml).

URL du fork : `https://github.com/jperchoc/ai-toolkit.git`

---

## Méthode 1 — re-pointer ton clone existant (recommandé)

Tu réutilises le **même dossier**, donc tu gardes ton venv, tes configs locales et
le cache Hugging Face. On garde aussi ostris comme `upstream` pour pouvoir se
resynchroniser plus tard.

```bash
cd /chemin/vers/ton/ai-toolkit          # ton clone ostris actuel

# 1. Mets de côté tes éventuelles modifs locales
git status
git stash            # ou git commit -am "wip" — à ne pas perdre

# 2. ostris devient "upstream", ton fork devient "origin"
git remote rename origin upstream
git remote add origin https://github.com/jperchoc/ai-toolkit.git

# 3. Récupère le fork et bascule main dessus
git fetch origin
git checkout -B main origin/main         # main pointe désormais sur le fork
git branch -u origin/main main           # main suit origin/main

# 4. (optionnel) réapplique tes modifs mises de côté
git stash pop
```

Vérifie :

```bash
git remote -v
# origin    https://github.com/jperchoc/ai-toolkit.git (fetch/push)
# upstream  https://github.com/ostris/ai-toolkit.git   (fetch/push)

git log --oneline -5     # tu dois voir les commits du fork (sample loras, perf batch…)
```

> ⚠️ `git checkout -B main origin/main` **écrase** l'état de ta branche `main`
> locale par celle du fork. C'est voulu (on veut exactement le fork), mais assure-toi
> d'avoir stashé/commité ton travail avant (étape 1).

---

## Méthode 2 — clone neuf à côté

Si tu préfères ne pas toucher à ton dossier ostris :

```bash
git clone https://github.com/jperchoc/ai-toolkit.git ai-toolkit-fork
cd ai-toolkit-fork
git remote add upstream https://github.com/ostris/ai-toolkit.git
```

Tu devras recréer le venv (`python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`).
Le cache Hugging Face (`~/.cache/huggingface`) est **partagé** entre les deux clones,
donc pas de re-téléchargement des modèles.

---

## Dépendances

Le fork **n'ajoute aucune dépendance** : `paged_adamw8bit` utilise `bitsandbytes`
(déjà requis), `torch.compile` et TF32 sont dans torch. Donc avec la Méthode 1,
**pas besoin de réinstaller** quoi que ce soit.

Si tu utilises l'UI, rebuild-la après bascule (`build_and_start` fait déjà le
`npm install` + build) :

```bash
cd ui
npm run build_and_start
```

---

## Se resynchroniser avec ostris plus tard

Pour récupérer les nouveautés d'ostris tout en gardant tes ajouts :

```bash
git checkout main
git fetch upstream
git merge upstream/main     # (ou: git rebase upstream/main)
# résous les éventuels conflits, puis
git push origin main
```

Le `merge` est plus sûr qu'un `rebase` sur un fork qui a déjà des commits publics.

---

## Revenir au repo officiel

Avec la Méthode 1, il suffit de rebasculer sur upstream :

```bash
git checkout -B main upstream/main
git branch -u upstream/main main
```

(ou refaire de `upstream` l'`origin` via `git remote rename`).

---

## Démarrage rapide krea2 12 Go

```bash
# édite name_or_path (ton raw local), le dataset, et offline si besoin
cp config/examples/train_lora_krea2_12gb.yaml config/mon_lora.yaml
python run.py config/mon_lora.yaml
```

Surveille dans les logs `[auto-offload] … offloading X%` et `[vram] … peak=…GB`,
puis ajuste `layer_offloading_reserved_gb` (baisse-le tant que le pic reste sous
~11 Go pour aller plus vite). Voir les commentaires de la config pour `qtype:
convrot8`, `compile_transformer`, `paged_adamw8bit`, etc.
