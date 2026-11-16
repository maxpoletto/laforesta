# Calendario del ceduo

Per la gestione del ceduo non si considerano i volumi, ma solo il calendario degli interventi. L'algoritmo non necessita di dati dendrometrici o simulazioni di crescita.

Le particelle interessate sono quelle con Governo=Ceduo in bosco/data/particelle.csv.

## Regole

1. Una particella puo' essere utilizzata con frequenza non superiore al valore del "Parametro" in particelle.csv (valori standard: 12, 15, 18 e 25 anni). L'intervallo si misura dal primo sub-intervento di un ciclo al primo sub-intervento del ciclo successivo.
2. Se una particella supera i 10 ha di superficie, l'intervento e' suddiviso in `ceil(superficie / 10)` sub-interventi di pari superficie, distanziati di almeno due anni. La suddivisione in parti uguali mantiene ciascun sub-intervento sotto i 10 ha evitando piccoli resti.
3. Se due particelle sono adiacenti (secondo le colonne A e B di bosco/data/cedui-adiacenti.csv), non possono essere utilizzate nello stesso anno e devono essere distanziate di almeno due anni. Il vincolo si applica a tutti gli anni dei sub-interventi, non solo al primo. L'adiacenza e' a coppie, non transitiva (es. se A-B e B-C sono adiacenti, A e C non lo sono necessariamente).

## Algoritmo di schedulazione

L'algoritmo utilizza una coda di priorita':

1. Per ogni particella cedua, calcolare il primo anno ammissibile = max(anno_inizio, ultimo_taglio + parametro).
2. Inserire tutte le particelle in una coda di priorita' ordinata per (anno ammissibile, compresa, particella).
3. Estrarre la prossima particella dalla coda.
4. Programmare un ciclo di sub-interventi:
   - Suddividere la superficie in `ceil(superficie / 10)` lotti di pari superficie.
   - Per ogni lotto, trovare il primo anno che soddisfi: (a) >= anno ammissibile corrente, (b) >= anno del sub-intervento precedente + 2 (per il 2o lotto in poi), e (c) nessun conflitto di adiacenza con interventi gia' programmati.
5. Calcolare il prossimo anno ammissibile = anno del primo sub-intervento di questo ciclo + parametro. Se rientra nel periodo di pianificazione, reinserire la particella nella coda.
6. Ripetere finche' la coda e' vuota.

Le date degli ultimi interventi provengono da calendario-mannesi.csv. Le particelle senza storico hanno ultimo_taglio = 0, e risultano quindi immediatamente ammissibili. Per lo spareggio nella coda di priorita', si scelgono prima le particelle non adiacenti nella stessa compresa in ordine lessicografico, poi quelle in comprese diverse.

## Anno iniziale registrato

Il primo anno del calendario pianificato può coincidere con l'ultimo anno del
`calendario` storico, con una sovrapposizione di un solo anno. In questo anno
iniziale compaiono solo gli interventi registrati. Le righe di ceduo non
richiedono un valore nella colonna `Prelievo totale (m3)` perché la
pianificazione del ceduo usa solo le superfici.

Un intervento registrato copre l'intera particella se la superficie è al massimo
10 ha. La colonna facoltativa `Intervento completo` può contenere `si` per
registrare un intervento completo anche oltre i 10 ha. In tal caso si usa
l'intera superficie della particella, senza sub-interventi di continuazione, e
il ciclo successivo diventa ammissibile solo dopo l'intero turno. Ad esempio,
una particella di 13,5 ha completata nel 2026 con turno di 18 anni torna
ammissibile nel 2044; il nuovo ciclo segue la suddivisione ordinaria (6,75 ha
per sub-intervento). Il diagramma Gantt indica il ciclo registrato completo come
un unico intervento.

Se la colonna è assente, vuota o contiene `no`, per le particelle più grandi
resta la regola ordinaria: il record è il primo sub-intervento di un
nuovo ciclo, con superficie `area / ceil(area / 10)`. Il programma completa i
sub-interventi rimanenti a distanza di almeno due anni, mantenendo l'anno di
inizio registrato nelle note e nel Gantt. Il ciclo successivo è ammissibile
dall'anno registrato più il turno della particella. Ad esempio, una particella
di 25 ha registrata nel 2026 con turno di 12 anni viene utilizzata nel 2026,
2028, 2030, poi 2038, 2040, 2042, salvo rinvii per adiacenza e il limite
dell'anno finale richiesto.

Tutte le date registrate del ceduo, anche precedenti al periodo visualizzato,
vincolano le adiacenze future. Le date registrate restano fisse; i nuovi
interventi sono programmati nel primo anno valido rispetto agli interventi
registrati e generati. Una data candidata in conflitto viene rinviata e non
causa un errore di pianificazione. Se invece un intervento registrato
nell'anno iniziale è in conflitto con un altro intervento registrato,
l'elaborazione si interrompe con un errore che identifica particelle e anni,
poiché nessuna delle due date può essere spostata.

## Esempi

* Una particella isolata di 8 ha con parametro=12, ultimo taglio nel 2015. Viene utilizzata nel 2027 e poi nel 2039.

* Una particella isolata di 25 ha con parametro=12, ultimo taglio nel 2015. Viene suddivisa in 3 sub-interventi di ~8,33 ha ciascuno, programmati nel 2027, 2029 e 2031, e analogamente nel 2039, 2041 e 2043.

* Due particelle adiacenti di 8 ha con parametro=12, ultimo taglio nel 2015 (particella A) e 2016 (B). A viene programmata nel 2027 e 2039. B non puo' essere programmata nel 2028 perche' dista solo un anno dal 2027, quindi viene programmata nel 2029 e 2041.

* Due particelle adiacenti di 12 ha con parametro=12, ultimo taglio nel 2015 (A) e 2016 (B). A viene suddivisa in 2 sub-interventi di 6 ha ciascuno, programmati nel 2027, 2029, 2039 e 2041. B non puo' essere programmata nel 2028-2030. Viene spostata al 2031 e 2033, e poi al 2043 e 2045.

## Sintassi della direttiva

@@calendario_ceduo(particelle=FILE,calendario=FILE,adiacenze=FILE,anno_inizio=M,anno_fine=N)

Output: tabella con colonne Anno, Compresa, Particella, Superficie (ha), Note. La colonna "Note" riporta "Continuazione intervento AAAA" per il secondo e successivi sub-interventi, dove AAAA e' l'anno del primo sub-intervento del ciclo.

@@tabella_ceduo(particelle=FILE,calendario=FILE,adiacenze=FILE,anno_inizio=M,anno_fine=N)

Output: un grafico tipo Gantt che illustra la vita di ogni lotto di 50 matricine
rilasciate. Una riga per ogni particella ceduo (ordinate in ordine naturale per
compresa e poi particella), una barra per ogni sub-intervento. Ogni barra inizia
nell'anno del sub-intervento e termina `2 * parametro` anni dopo (le matricine
sono mantenute per due turni successivi e tagliate al secondo). Le particelle
oltre i 10 ha hanno piu' slot di sub-intervento e quindi righe piu' alte. Le
barre che si estendono oltre `anno_fine` sono indicate con una freccia verso
destra.
