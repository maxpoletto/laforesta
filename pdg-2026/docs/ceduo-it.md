# Calendario del ceduo

Per la gestione del ceduo non si considerano i volumi, ma solo il calendario degli interventi. L'algoritmo non necessita di dati dendrometrici o simulazioni di crescita.

Le particelle interessate sono quelle con Governo=Ceduo in bosco/data/particelle.csv.

## Regole

1. Una particella puo' essere utilizzata con frequenza non superiore al valore del "Parametro" in particelle.csv (valori standard: 12, 15, 18 e 25 anni). L'intervallo si misura dal primo sub-intervento di un ciclo al primo sub-intervento del ciclo successivo.
2. Se una particella supera i 10 ha di superficie, l'intervento deve essere suddiviso in sub-interventi di massimo 10 ha, distanziati di almeno due anni.
3. Se due particelle sono adiacenti (secondo le colonne A e B di bosco/data/cedui-adiacenti.csv), non possono essere utilizzate nello stesso anno e devono essere distanziate di almeno due anni. Il vincolo si applica a tutti gli anni dei sub-interventi, non solo al primo. L'adiacenza e' a coppie, non transitiva (es. se A-B e B-C sono adiacenti, A e C non lo sono necessariamente).

## Algoritmo di schedulazione

L'algoritmo utilizza una coda di priorita':

1. Per ogni particella cedua, calcolare il primo anno ammissibile = max(anno_inizio, ultimo_taglio + parametro).
2. Inserire tutte le particelle in una coda di priorita' ordinata per (anno ammissibile, compresa, particella).
3. Estrarre la prossima particella dalla coda.
4. Programmare un ciclo di sub-interventi:
   - Suddividere la superficie in lotti di massimo 10 ha.
   - Per ogni lotto, trovare il primo anno che soddisfi: (a) >= anno ammissibile corrente, (b) >= anno del sub-intervento precedente + 2 (per il 2o lotto in poi), e (c) nessun conflitto di adiacenza con interventi gia' programmati.
5. Calcolare il prossimo anno ammissibile = anno del primo sub-intervento di questo ciclo + parametro. Se rientra nel periodo di pianificazione, reinserire la particella nella coda.
6. Ripetere finche' la coda e' vuota.

Le date degli ultimi interventi provengono da calendario-mannesi.csv. Le particelle senza storico hanno ultimo_taglio = 0, e risultano quindi immediatamente ammissibili. Per lo spareggio nella coda di priorita', si scelgono prima le particelle non adiacenti nella stessa compresa in ordine lessicografico, poi quelle in comprese diverse.

## Esempi

* Una particella isolata di 8 ha con parametro=12, ultimo taglio nel 2015. Viene utilizzata nel 2027 e poi nel 2039.

* Una particella isolata di 25 ha con parametro=12, ultimo taglio nel 2015. Viene utilizzata nel 2027 (10 ha), 2029 (10 ha) e 2031 (5 ha), e analogamente nel 2039, 2041 e 2043.

* Due particelle adiacenti di 8 ha con parametro=12, ultimo taglio nel 2015 (particella A) e 2016 (B). A viene programmata nel 2027 e 2039. B non puo' essere programmata nel 2028 perche' dista solo un anno dal 2027, quindi viene programmata nel 2029 e 2041.

* Due particelle adiacenti di 12 ha con parametro=12, ultimo taglio nel 2015 (A) e 2016 (B). A viene programmata nel 2027 (10 ha), 2029 (2 ha), 2039 e 2041. B non puo' essere programmata nel 2028-2030. Viene spostata al 2031 e 2033, e poi al 2043 e 2045.

## Sintassi della direttiva

@@calendario_ceduo(particelle=FILE,calendario=FILE,adiacenze=FILE,anno_inizio=M,anno_fine=N)

Output: tabella con colonne Anno, Compresa, Particella, Superficie (ha), Note. La colonna "Note" riporta "Continuazione intervento AAAA" per il secondo e successivi sub-interventi, dove AAAA e' l'anno del primo sub-intervento del ciclo.
