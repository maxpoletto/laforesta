# acc.py

Strumento per l'analisi di accrescimenti forestali: calcolo altezze/volumi, curve ipsometriche, ecc.

## Modalità

### 1. Genera equazioni (curve di regressione altezza-diametro)
```bash
./acc.py --genera-equazioni --funzione={log,lin} --fonte-altezze={tabelle,ipsometro,originali} \
         --input FILE_INPUT --output FILE_EQUAZIONI --particelle METADATI_PARTICELLE
```

Interpola curve (y = a·ln(x) + b oppure y = a·x + b) per ogni coppia (compresa, genere).
Output CSV: `compresa,genere,funzione,a,b,r2,n`

### 2. Calcola altezze e volumi
```bash
./acc.py --calcola-altezze-volumi --equazioni FILE_EQUAZIONI --input FILE_INPUT --output FILE_OUTPUT
```

In un unico passaggio:
1. Applica le equazioni per stimare le altezze degli alberi (invariate se non esiste equazione)
2. Calcola il volume V(m³) per ogni albero usando le tavole di Tabacchi

Questo garantisce che altezze e volumi siano sempre coerenti.

### 3. Genera report
```bash
./acc.py --report --formato={html,latex,pdf} --dati DIR_DATI \
         --particelle METADATI_PARTICELLE --input FILE_TEMPLATE --output-dir PERCORSO
```

Elabora il template, sostituendo le `@@direttive` con grafici/tabelle. La modalità PDF esegue pdflatex.
Ogni direttiva specifica i propri file dati tramite i parametri `alberi=` e `equazioni=` (relativi a `--dati`).

### 4. Lista particelle
```bash
./acc.py --lista-particelle --particelle METADATI_PARTICELLE
```

Elenca tutte le tuple (compresa, particella).

## Direttive per Template

- `@@cd(parametri)` — Istogramma delle classi diametriche con metadati
- `@@ci(parametri)` — Grafico a dispersione altezza-diametro con curve di regressione
- `@@tsv(parametri)` — Tabella volumi con intervalli di fiducia opzionali
- `@@tpt(parametri)` — Tabella prelievo totale basata su regole volume/età

### Parametri Comuni

| Parametro | Valori | Descrizione | Obbligatorio | Applicabile a |
|-----------|--------|-------------|--------------|---------------|
| `alberi=FILE` | nome file | CSV dati alberi (relativo a `--dati`) | **Sì** | tutti |
| `equazioni=FILE` | nome file | CSV equazioni (relativo a `--dati`) | **Sì** per `@@ci` | solo `@@ci` |
| `compresa=NOME` | nome compresa | Filtra per compresa (default: tutte) | No | tutti |
| `particella=NOME` | nome particella | Filtra per particella (richiede compresa) | No | tutti |
| `genere=GENERE` | nome specie | Filtra per specie (default: tutte) | No | tutti |
| `per_compresa` | `si`, `no` | Raggruppa per compresa (default: `si`) | No | `@@tsv`, `@@tpt` |
| `per_particella` | `si`, `no` | Raggruppa per particella (default: `si`) | No | `@@tsv`, `@@tpt` |
| `per_genere` | `si`, `no` | Raggruppa per genere (default: `si`) | No | `@@tsv`, `@@tpt` |
| `totali` | `si`, `no` | Aggiungi riga totali (default: `no`) | No | `@@tsv`, `@@tpt` |

### Parametri `@@tsv`

| Parametro | Valori | Descrizione |
|-----------|--------|-------------|
| `stime_totali` | `si`, `no` | Mostra volumi totali stimati (default: `no`) |
| `intervallo_fiduciario` | `si`, `no` | Mostra intervalli di fiducia (default: `no`) |

### Parametri `@@tpt`

| Parametro | Valori | Descrizione | Obbligatorio |
|-----------|--------|-------------|--------------|
| `comparti=FILE` | nome file | CSV regole comparti (relativo a `--dati`) | **Sì** |
| `provv_vol=FILE` | nome file | CSV regole prelievo basate su volume | **Sì** |
| `provv_eta=FILE` | nome file | CSV regole prelievo basate su età | **Sì** |
| `comparto` | `si`, `no` | Mostra colonna comparto (default: `si`) | No |
| `col_volume` | `si`, `no` | Mostra colonna volume totale (default: `no`) | No |
| `col_pp_max` | `si`, `no` | Mostra colonna PP_max % (default: `no`) | No |
| `col_prel_ha` | `si`, `no` | Mostra prelievo per ettaro (default: `si`) | No |
| `col_prel_tot` | `si`, `no` | Mostra prelievo totale (default: `si`) | No |

**Algoritmo prelievo** (per particella):
1. Poni v = volume per ettaro totale, per tutte le specie
2. Trova provvigione_minima (pm) dal comparto
3. Trova PP_max dalle regole volume: prima riga dove v > PPM × pm / 100
4. Limita PP_max usando regole età: prima riga dove età_media > Anni
5. Prelievo per specie per ettaro = volume per ettaro della specie × PP_max / 100

Le particelle a ceduo (Comparto F) sono escluse dai calcoli di prelievo.

**Parametri multivalore**: `alberi`, `equazioni`, `compresa`, `particella` e `genere` possono essere ripetuti:
```
@@cd(alberi=alberi1.csv, alberi=alberi2.csv, compresa=Serra, compresa=Fabrizia)
@@ci(alberi=alberi.csv, equazioni=eq1.csv, equazioni=eq2.csv, compresa=Serra)
```
I file `alberi`/`equazioni` multipli vengono concatenati; i filtri multipli sono combinati con OR.

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

- **Curve di regressione** nei grafici `@@ci` usano `FILE_EQUAZIONI` (non ricalcolate dai dati correnti) per riflettere la qualità dell'interpolazione originale
- **Intervalli di fiducia volumi**: Aggregazione conservativa (somma dei margini) per specie miste
