# pdg.py

Strumento per l'analisi di accrescimenti forestali: calcolo altezze/volumi, curve ipsometriche, ecc.

## Modalità

### 1. Genera equazioni (curve di regressione altezza-diametro)
```bash
./pdg.py --genera-equazioni --funzione={log,lin} --fonte-altezze={tabelle,ipsometro,originali} \
         --input FILE_INPUT --output FILE_EQUAZIONI --particelle METADATI_PARTICELLE
```

Interpola curve (y = a·ln(x) + b oppure y = a·x + b) per ogni coppia (compresa, genere).
Output CSV: `compresa,genere,funzione,a,b,r2,n`

### 2. Calcola altezze e volumi
```bash
./pdg.py --calcola-altezze-volumi --equazioni FILE_EQUAZIONI --input FILE_INPUT --output FILE_OUTPUT \
         [--coeff-pressler VALORE]
```

In un unico passaggio:
1. Applica le equazioni per stimare le altezze degli alberi (invariate se non esiste equazione)
2. Calcola il volume V(m³) per ogni albero usando le tavole di Tabacchi

`--coeff-pressler` imposta il coefficiente di Pressler per tutti gli alberi.

### 3. Calcola incrementi
```bash
./pdg.py --calcola-incrementi --input FILE_INPUT --output FILE_OUTPUT
```

Calcola l'incremento percentuale (IP) per ogni albero e scrive il risultato nel CSV di output.

### 4. Genera report
```bash
./pdg.py --report --formato={csv,html,tex,pdf} --dati DIR_DATI \
         --particelle METADATI_PARTICELLE --input FILE_TEMPLATE --output-dir PERCORSO \
         [--ometti-generi-sconosciuti] [--non-rigenerare-grafici] \
         [--separatore-decimale {punto,virgola}]
```

Elabora il template, sostituendo le `@@direttive` con grafici/tabelle. La modalità PDF esegue pdflatex.
Ogni direttiva specifica i propri file dati tramite i parametri `alberi=` e `equazioni=` (relativi a `--dati`).

Opzioni:
- `--ometti-generi-sconosciuti`: ometti dai grafici i generi senza equazioni
- `--non-rigenerare-grafici`: non rigenerare file grafici già esistenti
- `--separatore-decimale`: separatore decimale nell'output (`virgola` = stile italiano, default)

### 5. Lista particelle
```bash
./pdg.py --lista-particelle --particelle METADATI_PARTICELLE
```

Elenca tutte le tuple (compresa, particella).

## Direttive per Template

### Grafici
- `@@grafico_classi_diametriche(parametri)` — Istogramma delle classi diametriche
- `@@grafico_classi_ipsometriche(parametri)` — Grafico a dispersione altezza-diametro con curve di regressione
- `@@grafico_incremento_percentuale(parametri)` — Grafico incremento percentuale per classe diametrica

### Tabelle
- `@@volumi(parametri)` — Tabella volumi con intervalli di fiducia opzionali
- `@@tabella_classi_diametriche(parametri)` — Tabella classi diametriche
- `@@tabella_incremento_percentuale(parametri)` — Tabella incremento percentuale
- `@@prelievi(parametri)` — Tabella prelievi basata su regole comparto/età
- `@@piano_di_taglio(parametri)` — Piano di taglio pluriennale (simulazione del calendario prelievi)

### Strutturali
- `@@particelle(compresa=X, modello=BASENAME)` — Espande un sotto-template per ogni particella della compresa. Il sotto-template può usare `@@compresa` e `@@particella` come segnaposto, e contenere ulteriori direttive.
- `@@prop(compresa=X, particella=Y)` — Inserisce le proprietà (metadati) della particella

### Parametri Comuni

| Parametro | Valori | Descrizione | Obbligatorio |
|-----------|--------|-------------|--------------|
| `alberi=FILE` | nome file | CSV dati alberi (relativo a `--dati`) | **Sì** (tutti tranne `@@prop`, `@@particelle`) |
| `equazioni=FILE` | nome file | CSV equazioni (relativo a `--dati`) | **Sì** per `@@grafico_classi_ipsometriche` |
| `compresa=NOME` | nome compresa | Filtra per compresa (default: tutte) | No |
| `particella=NOME` | nome particella | Filtra per particella (richiede compresa) | No |
| `genere=GENERE` | nome specie | Filtra per specie (default: tutte) | No |
| `per_compresa` | `si`, `no` | Raggruppa per compresa (default: `si`) | No |
| `per_particella` | `si`, `no` | Raggruppa per particella (default: `si`) | No |
| `per_genere` | `si`, `no` | Raggruppa per genere (default varia per direttiva) | No |
| `totali` | `si`, `no` | Aggiungi riga totali (default: `no`) | No |
| `stime_totali` | `si`, `no` | Usa stime totali scalate all'area della particella (default: `si`) | No |

**Parametri multivalore**: `alberi`, `equazioni`, `compresa`, `particella` e `genere` possono essere ripetuti:
```
@@grafico_classi_diametriche(alberi=alberi1.csv, alberi=alberi2.csv, compresa=Serra, compresa=Fabrizia)
@@grafico_classi_ipsometriche(alberi=alberi.csv, equazioni=eq1.csv, equazioni=eq2.csv, compresa=Serra)
```
I file `alberi`/`equazioni` multipli vengono concatenati; i filtri multipli sono combinati con OR.

### Parametri Grafici

| Parametro | Valori | Descrizione |
|-----------|--------|-------------|
| `metrica` | vedi sotto | Metrica da rappresentare (default varia per direttiva) |
| `stile` | testo libero | Classe CSS (HTML) o opzioni `\includegraphics` (LaTeX) |
| `x_max` | intero | Impone il massimo dell'asse x (0 = automatico) |
| `y_max` | intero | Impone il massimo dell'asse y (0 = automatico) |

Metriche `@@grafico_classi_diametriche`: `alberi_ha`, `G_ha`, `volume_ha`, `alberi_tot`, `G_tot`, `volume_tot`, `altezza` (default: `alberi_ha`).

Metriche `@@grafico_incremento_percentuale`: `ip`, `ic` (default: `ip`). `per_compresa` e `per_particella` sono `no` per default.

### Parametri `@@volumi`

| Parametro | Valori | Descrizione |
|-----------|--------|-------------|
| `intervallo_fiduciario` | `si`, `no` | Mostra intervalli di fiducia (default: `no`) |
| `solo_mature` | `si`, `no` | Solo alberi maturi (D > 20 cm) (default: `no`) |

### Parametri `@@prelievi`

Le regole di prelievo sono definite in `harvest_rules.py` (limiti per comparto basati su tavole di volume ed età).

| Parametro | Valori | Descrizione |
|-----------|--------|-------------|
| `col_comparto` | `si`, `no` | Mostra colonna comparto (default: `si`) |
| `col_eta` | `si`, `no` | Mostra colonna età media (default: `si`) |
| `col_area_ha` | `si`, `no` | Mostra area in ettari (default: `si`) |
| `col_volume` | `si`, `no` | Mostra volume totale (default: `no`) |
| `col_volume_ha` | `si`, `no` | Mostra volume per ettaro (default: `no`) |
| `col_volume_mature` | `si`, `no` | Mostra volume alberi maturi (default: `si`) |
| `col_volume_mature_ha` | `si`, `no` | Mostra volume alberi maturi per ettaro (default: `si`) |
| `col_pp_max` | `si`, `no` | Mostra PP_max % (default: `si`) |
| `col_prelievo_ha` | `si`, `no` | Mostra prelievo per ettaro (default: `si`) |
| `col_prelievo` | `si`, `no` | Mostra prelievo totale (default: `si`) |

Nota: il filtro `genere` non è ammesso — usare `per_genere=si` per raggruppare per specie.

### Parametri `@@piano_di_taglio`

| Parametro | Valori | Descrizione | Obbligatorio |
|-----------|--------|-------------|--------------|
| `volume_obiettivo` | numero | Volume in piedi obiettivo (m³/ha) | **Sì** |
| `anno_inizio` | anno | Primo anno di taglio (default: 2026) | No |
| `anno_fine` | anno | Ultimo anno di taglio (default: 2040) | No |
| `intervallo` | anni | Intervallo tra i tagli (default: 10) | No |
| `mortalita` | frazione | Tasso di mortalità annua (default: 0) | No |
| `calendario=FILE` | nome file | CSV tagli passati (relativo a `--dati`) | No |
| `col_comparto` | `si`, `no` | Mostra colonna comparto (default: `si`) | No |
| `col_eta` | `si`, `no` | Mostra colonna età media (default: `si`) | No |
| `col_pp_max` | `si`, `no` | Mostra PP_max % (default: `si`) | No |
| `col_prima_dopo` | `si`, `no` | Mostra volumi prima/dopo (default: `si`, richiede `per_particella=si`) | No |

## Formati File

- **File equazioni** (CSV): Coefficienti di regressione per le relazioni altezza-diametro
  - Colonne: `compresa,genere,funzione,a,b,r2,n`
  - Una riga per coppia (compresa, genere) con dati sufficienti (n ≥ 10)

- **File altezze** (CSV): Misurazioni sul campo da ipsometro o dati tabulari
  - Ipsometro: `Compresa,Particella,Area saggio,Genere,D(cm),h(m)`
  - Tabelle: `Genere,Classe diametrica,Altezza indicativa`

- **Database alberi** (CSV): Inventario completo con attributi calcolati
  - Colonne: `Compresa,Particella,Area saggio,Genere,D(cm),h(m),V(m³),Fustaia,Classe diametrica`
  - Altezze inizialmente stimate, affinate usando le equazioni
  - Volumi calcolati usando le tavole di Tabacchi

## Note Implementative

- **Curve di regressione** nei grafici `@@grafico_classi_ipsometriche` usano il file equazioni (non ricalcolate dai dati correnti) per riflettere la qualità dell'interpolazione originale
- **Intervalli di fiducia volumi**: Aggregazione conservativa (somma dei margini) per specie miste
- **Regole di prelievo** sono codificate in `harvest_rules.py` come funzione di comparto, età, volume e area basimetrica
