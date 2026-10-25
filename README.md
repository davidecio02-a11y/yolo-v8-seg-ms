# Segmentazione dei difetti nel calcestruzzo armato

**Confronto tra YOLOv8s-seg e una pipeline semantica compatta ispirata a MS-YOLOv8-Seg, sul dataset S2DS.**

Il progetto studia la segmentazione automatica di aree di **corrosione e spalling** nelle immagini di superfici in calcestruzzo. Le due tipologie di difetto vengono unite in una sola classe binaria: *difetto* contro *sfondo*. Il lavoro confronta YOLOv8s-seg con diverse configurazioni di una rete di segmentazione semantica, valutandone accuratezza, falsi positivi e costo computazionale.

> **Nota sull'architettura:** il modello personalizzato è ispirato al paper MS-YOLOv8-Seg, ma non ne costituisce una replica integrale. In particolare, viene mantenuta la struttura generale del neck YOLOv8, modificando alcuni blocchi interni; la testa originale di segmentazione di istanza viene sostituita da una testa semantica che produce una singola maschera binaria.

## Indice

- [Obiettivi](#obiettivi)
- [Dataset e preelaborazione](#dataset-e-preelaborazione)
- [Modelli e configurazioni](#modelli-e-configurazioni)
- [Requisiti](#requisiti)
- [Organizzazione dei dati](#organizzazione-dei-dati)
- [Esecuzione degli esperimenti](#esecuzione-degli-esperimenti)
- [Valutazione](#valutazione)
- [Risultati principali](#risultati-principali)
- [Efficienza computazionale](#efficienza-computazionale)
- [Output generati](#output-generati)
- [Applicazione e sviluppi futuri](#applicazione-e-sviluppi-futuri)
- [Limiti e riproducibilità](#limiti-e-riproducibilità)
- [Riferimenti](#riferimenti)

## Obiettivi

1. Confrontare una baseline YOLOv8s-seg con modelli semantici personalizzati.
2. Verificare il contributo dei moduli Ghost, RepBlock e SimAM e della loss DSIoU.
3. Misurare non solo la qualità della maschera, ma anche i falsi allarmi sulle tessere prive di difetti.
4. Studiare il compromesso tra accuratezza, dimensioni del modello e latenza di inferenza.
5. Realizzare una base sperimentale per un prototipo di supporto all'ispezione di superfici in calcestruzzo.

## Dataset e preelaborazione

Il progetto usa il dataset **S2DS**, composto da 743 immagini originali di risoluzione 1024 × 1024 pixel, suddivise in training, validation e test. Le immagini vengono suddivise in tessere (*patch*) da 256 × 256 pixel.

La preparazione descritta negli esperimenti produce:

| Split | Tessere positive | Tessere negative | Totale |
|---|---:|---:|---:|
| Training | 1.416 | 708 | 2.124 |
| Validation | 164 | 82 | 246 |
| Test | 187 | 187 | 374 |

Le tessere positive contengono almeno un pixel di difetto; quelle negative non contengono difetti nella maschera di riferimento. Il test comprende 90 immagini sorgente. La suddivisione viene effettuata a livello di immagine sorgente per evitare che tessere provenienti dalla stessa immagine originale finiscano in split differenti.

Corrosione e spalling sono fuse nella classe binaria `corrosion_spalling`. Le annotazioni possono essere ricavate da poligoni YOLO in file `.txt` oppure da maschere `.png` in scala di grigi, purché i file corrispondano alle immagini tramite lo stesso nome base.

**Licenza del dataset:** S2DS è utilizzato per finalità accademiche secondo le condizioni indicate dal relativo distributore. Non ridistribuire immagini o annotazioni del dataset senza averne verificato e ottenuto i diritti necessari.

## Modelli e configurazioni

### YOLOv8s-seg

Modello di riferimento per la segmentazione di istanze. Le predizioni di maschera sono unite a livello di pixel per essere confrontate con la maschera binaria usata nel progetto.

### YOLO-MS semantico

Il modello personalizzato utilizza parti preaddestrate di YOLOv8-Seg come inizializzazione, ma sostituisce la testa originale con una testa semantica che restituisce un solo canale di logits per pixel. Dopo la soglia, l'output è una maschera binaria per corrosione e spalling.

Le modifiche implementate sono specifiche del codice del progetto:

- **GhostBottleneck:** sostituisce i bottleneck interni ai blocchi C2f quando la variante Ghost è attiva.
- **GhostDown:** sostituisce alcune convoluzioni di downsampling nel neck.
- **RepBlock:** blocco convoluzionale multi-ramo usato nella testa semantica quando abilitato; i rami possono essere riparametrizzati per l'inferenza.
- **SimAM:** modulo di attenzione senza parametri apprendibili aggiuntivi, inserito nella testa semantica quando abilitato.
- **DSIoU:** variante della loss che pesa maggiormente le zone vicine ai bordi della maschera; viene applicata durante il training, non è un modulo di inferenza.

Il neck RepGFPN-Lite descritto nel paper di riferimento **non è implementato integralmente**: il codice conserva la struttura generale YOLOv8 e ne modifica alcune operazioni interne.

### Configurazioni sperimentali

| Nome | Configurazione |
|---|---|
| `yolo_pre` | YOLOv8s-seg con pesi preaddestrati |
| `yolo_scratch` | YOLOv8s-seg addestrato da zero |
| `ms_base` | Testa semantica personalizzata, senza Ghost, RepBlock o SimAM, loss base |
| `ms_ghost` | `ms_base` con moduli Ghost attivi |
| `ms_rep` | `ms_base` con RepBlock attivo nella testa |
| `ms_simam` | `ms_base` con SimAM attivo nella testa |
| `ms_dsiou` | `ms_base` con loss DSIoU |
| `ms_full` | Ghost + RepBlock + SimAM + DSIoU |
| `ms_base_scratch` | `ms_base` senza inizializzazione preaddestrata |
| `ms_full_scratch` | `ms_full` senza inizializzazione preaddestrata |

## Requisiti

- Python 3.10 o versione compatibile con le versioni di PyTorch e Ultralytics utilizzate.
- PyTorch e, per l'addestramento accelerato, una versione CUDA compatibile con la GPU disponibile.
- Ultralytics YOLO.
- OpenCV (`opencv-python`), NumPy, pandas e SciPy.
- Matplotlib per la generazione dei grafici del report.
- Streamlit solo per l'interfaccia applicativa, se si utilizza il relativo file nel progetto.

Si consiglia di creare un ambiente virtuale. Installare prima **PyTorch/torchvision nella build appropriata per il proprio sistema** e poi le altre dipendenze:

```bash
python -m venv .venv
```

Su Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Installazione delle librerie utilizzate dallo script di training e analisi:

```bash
pip install ultralytics opencv-python numpy pandas scipy matplotlib
```

Verificare che `torch` sia installato e che `torch.cuda.is_available()` restituisca `True` quando si desidera utilizzare la GPU. Le versioni esatte vanno mantenute coerenti con l'ambiente in cui sono state eseguite le prove originali.

## Organizzazione dei dati

Lo script si aspetta una struttura simile alla seguente. Il nome dello split di validazione può essere `valid` oppure `val`.

```text
spalling_corrosion_patches_v2_new/
├── train/
│   ├── images/
│   └── labels/
├── valid/                 # oppure val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Le immagini possono essere in formato PNG, JPG, JPEG o BMP. Le label devono essere coerenti con le immagini corrispondenti. Per le annotazioni poligonali in formato YOLO, la classe e le coordinate normalizzate devono essere presenti nel file `.txt`; in alternativa, è possibile usare una maschera `.png` in scala di grigi.

Nel codice di orchestrazione i percorsi predefiniti sono:

```text
Dataset:       C:\spalling_corrosion_patches_v2_new
Cartella run:  D:\yolo_seg\final_runs
Pesi iniziali: yolov8s-seg.pt
```

È possibile specificare percorsi diversi con gli argomenti `--root` e `--work`, senza modificare il codice.

## Esecuzione degli esperimenti

Gli esempi seguenti assumono che lo script di orchestrazione sia salvato come `final_project.py` nella cartella di lavoro. Eseguire i comandi dalla cartella che contiene lo script.

### 1. Controllare il piano di esecuzione

Il comando mostra le configurazioni e i seed ancora da eseguire senza avviare il training:

```bash
python final_project.py --plan --root "C:\percorso\dataset" --work "D:\yolo_seg\final_runs"
```

### 2. Eseguire la pipeline completa

```bash
python final_project.py --stage all --root "C:\percorso\dataset" --work "D:\yolo_seg\final_runs"
```

La pipeline gestisce le run previste, valuta i checkpoint disponibili, misura l'efficienza quando richiesto e genera il report. I risultati già salvati vengono riutilizzati secondo le regole dello script.

### 3. Eseguire solo alcune configurazioni o seed

Esempio con tre seed per il confronto tra YOLO e due varianti MS:

```bash
python final_project.py --stage all --configs yolo_pre,ms_base,ms_full --seeds 0,1,2 --root "C:\percorso\dataset" --work "D:\yolo_seg\final_runs"
```

Altri stadi disponibili:

```bash
python final_project.py --stage train
python final_project.py --stage eval
python final_project.py --stage efficiency
python final_project.py --stage report
```

- `train`: addestra le configurazioni selezionate.
- `eval`: valuta checkpoint già esistenti.
- `efficiency`: misura parametri, GFLOPs e latenze.
- `report`: aggrega i CSV di valutazione disponibili e produce tabelle e grafici.

`--configs` accetta nomi separati da virgole; `--seeds` accetta numeri separati da virgole. Per conoscere le opzioni disponibili:

```bash
python final_project.py --help
```

### 4. Prova rapida

Per verificare che ambiente, percorsi e pipeline funzionino, si può usare:

```bash
python final_project.py --quick --stage all --root "C:\percorso\dataset"
```

La modalità rapida usa poche epoche, un solo seed e un sottoinsieme delle configurazioni; **i suoi risultati sono destinati al controllo della pipeline, non alla valutazione finale**. Per impostazione predefinita usa una cartella di lavoro separata con suffisso `_quick`.

### Durata e ripresa

Gli addestramenti completi possono richiedere molto tempo, soprattutto per le varianti da zero. `--budget-min` permette allo script di fermarsi tra due run dopo un budget in minuti; rilanciando lo stesso comando è possibile riprendere il lavoro usando gli artefatti già salvati. Gli errori vengono registrati in `errors.log` nella cartella di lavoro.

## Protocollo di training e transfer learning

I parametri principali configurati nello script includono input da 256 × 256, batch size 16 e seed predefiniti `0, 1, 2` per le prove principali.

Per il modello semantico con pretraining, la procedura prevede una fase iniziale di warm-up di 5 epoche nella quale vengono addestrati i parametri nuovi mentre i parametri classificati come preaddestrati sono congelati; segue una fase di fine-tuning di 45 epoche con i gruppi di parametri e learning rate previsti dal codice. Le configurazioni semantiche da zero sono addestrate per 120 epoche. Per YOLO, lo script prevede 50 epoche con pesi preaddestrati e 120 da zero.

Il transfer learning non significa che ogni parametro della rete finale provenga da COCO. Le parti comuni della rete YOLO vengono inizializzate dai pesi di `yolov8s-seg.pt`; i moduli sostitutivi Ghost e la testa semantica personalizzata vengono in larga parte inizializzati da zero. Per `GhostDown`, il codice copia alcuni pesi e statistiche del ramo convoluzionale principale, mentre il ramo economico aggiuntivo è nuovo. SimAM non ha parametri apprendibili. La loss DSIoU non trasferisce parametri: cambia la funzione obiettivo durante il training.

## Valutazione

La soglia di segmentazione e il filtro delle componenti connesse piccole vengono scelti sulla validation; il test viene mantenuto per la valutazione finale. Il protocollo riporta diverse metriche perché nessuna, da sola, descrive completamente il comportamento di un modello.

- **IoU micro complessiva:** sovrapposizione pixel-level su tutte le tessere, incluse quelle negative.
- **IoU sui tile positivi:** IoU aggregata considerando solo le tessere con difetti reali.
- **IoU media per tile positivo:** media delle IoU calcolate separatamente su ciascuna tessera positiva; dà peso uguale a ogni tessera.
- **Dice, precisione e recall:** descrivono rispettivamente la sovrapposizione, la quota di pixel predetti come difetto che è corretta e la quota di pixel difettosi individuata.
- **FP neg >1%:** percentuale di tessere negative nelle quali l'area predetta come difetto supera l'1% della tessera.
- **FP neg qualsiasi pixel:** percentuale di tessere negative che contengono almeno un pixel falso positivo.
- **AUC a livello di tessera:** usa la frazione di area predetta come punteggio per distinguere tessere positive e negative; non è un'AUC pixel-level e non misura la calibrazione delle probabilità.
- **Accuratezza bilanciata a livello di tessera:** media tra sensibilità e specificità usando l'area predetta come criterio per classificare una tessera.
- **Bootstrap appaiato:** stima l'incertezza delle differenze tra modelli ricampionando le immagini sorgente. Non sostituisce la variabilità tra seed.

Le metriche riportate in percentuale non devono essere confuse con punti percentuali di differenza. Per esempio, passare dal 86% al 88% equivale a +2 punti percentuali.

## Risultati principali

La tabella riporta la media ± deviazione standard su tre seed (0, 1, 2), secondo i risultati del progetto.

| Configurazione | IoU complessiva (%) | IoU tile positivi (%) | IoU media tile positivi (%) | Tile negativi con FP >1% (%) | AUC tessera (%) |
|---|---:|---:|---:|---:|---:|
| `yolo_pre` | 78,5 ± 2,8 | 86,5 ± 0,7 | 73,9 ± 3,8 | 10,9 ± 1,5 | 90,5 ± 3,3 |
| `ms_base` | 86,2 ± 0,9 | 88,5 ± 1,0 | 78,7 ± 1,6 | 11,8 ± 2,1 | 96,9 ± 0,3 |
| `ms_ghost` | 81,8 ± 2,0 | 85,6 ± 1,3 | 74,9 ± 1,7 | 19,4 ± 2,6 | 95,9 ± 0,7 |
| `ms_rep` | 85,9 ± 1,7 | 89,4 ± 0,8 | 80,6 ± 0,8 | 13,0 ± 3,6 | 97,1 ± 1,5 |
| `ms_simam` | 86,9 ± 0,3 | 89,2 ± 0,8 | 79,5 ± 1,3 | 11,2 ± 2,1 | 97,4 ± 0,8 |
| `ms_dsiou` | 86,9 ± 0,7 | 88,9 ± 0,5 | 79,5 ± 0,3 | 13,5 ± 1,7 | 97,7 ± 0,5 |
| `ms_full` | 80,8 ± 0,8 | 85,2 ± 0,2 | 74,8 ± 0,6 | 22,5 ± 3,2 | 96,0 ± 0,9 |
| `yolo_scratch` | 75,8 ± 3,9 | 83,0 ± 1,5 | 70,2 ± 3,5 | 10,9 ± 6,8 | 89,2 ± 0,4 |
| `ms_base_scratch` | 77,9 ± 2,5 | 82,2 ± 2,2 | 72,8 ± 1,6 | 24,4 ± 2,4 | 95,7 ± 0,8 |
| `ms_full_scratch` | 77,5 ± 0,6 | 84,2 ± 0,4 | 74,8 ± 1,3 | 31,6 ± 1,9 | 93,9 ± 0,7 |

### Interpretazione sintetica

- `ms_base`, `ms_simam` e `ms_dsiou` sono configurazioni promettenti secondo l'IoU complessiva.
- `ms_rep` ottiene la più alta IoU sui tile positivi e la più alta IoU media per tile positivo nella tabella, ma presenta una variabilità maggiore su alcune metriche.
- `ms_ghost` e soprattutto `ms_full` mostrano un aumento dei falsi allarmi oltre l'1% dell'area rispetto a `ms_base`.
- Il pretraining migliora sensibilmente la IoU sui difetti per `ms_base` rispetto alla corrispondente variante da zero. L'effetto varia in base alla configurazione.
- Le differenze vanno interpretate con gli intervalli di confidenza e con la limitata numerosità dei seed. Non è corretto affermare una superiorità statisticamente dimostrata quando l'intervallo della differenza comprende lo zero.

## Efficienza computazionale

Le misure riportate sotto sono state ottenute su una **RTX 5050 Laptop GPU** con input da 256 × 256; le latenze dipendono dall'hardware, dal software e dalla modalità di misurazione. Non vanno considerate misure su dispositivo embedded.

| Modello | Parametri (M) | GFLOPs @256 | GPU batch 1 (ms) | GPU throughput batch 16 (immagini/s) | CPU 1 thread (ms) |
|---|---:|---:|---:|---:|---:|
| YOLOv8s-seg | 11,81 | 6,41 | 8,9 | 958 | 96 |
| `ms_base` | 9,66 | 5,97 | 6,9 | 823 | 97 |
| `ms_rep` | 9,66 | 5,97 | 5,8 | 836 | 97 |
| `ms_ghost` | 5,76 | 4,50 | 8,5 | 903 | 70 |
| `ms_full` | 5,76 | 4,50 | 8,8 | 887 | 85 |

Le varianti Ghost riducono i parametri di circa il 51% e i GFLOPs di circa il 30% rispetto a YOLOv8s-seg, ma non risultano più rapide in tutti gli scenari GPU. La possibile distribuzione su dispositivi embedded è un obiettivo futuro che richiede misure reali sul dispositivo di destinazione, oltre a verifiche su memoria, consumi e latenza end-to-end.

## Output generati

Con la cartella di lavoro predefinita `D:\yolo_seg\final_runs`, lo script organizza gli artefatti in modo simile a questo:

```text
final_runs/
├── ckpt/                 # checkpoint dei modelli
├── csv/                  # metriche per configurazione e seed
├── report/
│   ├── runs.csv          # risultati aggregati per run
│   ├── report.md         # report testuale/Markdown
│   ├── fig_ablation.png  # grafico dell'ablation
│   └── fig_accuratezza_vs_costo.png
├── efficiency.csv        # misure di complessità e latenza
├── errors.log            # dettagli delle run fallite, se presenti
└── data_runtime.yaml     # YAML generato per l'uso con Ultralytics
```

I file vengono creati in base agli stadi effettivamente eseguiti e alla disponibilità dei risultati; non tutti saranno presenti prima del completamento della pipeline.

## Applicazione e sviluppi futuri

Il progetto prevede un prototipo applicativo Streamlit per caricare immagini o video, eseguire la segmentazione a tessere e visualizzare le aree sospette. L'applicazione è pensata come **supporto all'ispezione**: le segnalazioni automatiche devono essere verificate da un tecnico.

Sviluppi futuri possibili comprendono:

- test dell'applicazione su hardware embedded concreto;
- valutazione su ulteriori superfici, condizioni di luce e immagini acquisite sul campo;
- analisi più ampia dei falsi positivi su superfici sane;
- integrazione futura con altre modalità di acquisizione o con strumenti di documentazione tecnica.

Per avviare Streamlit, usare il nome effettivo del file applicativo presente nel repository, ad esempio `streamlit run app.py` se il file si chiama `app.py`.

## Limiti e riproducibilità

- Gli esperimenti sono condotti su un solo dataset e su un insieme di test con rapporto 1:1 tra tessere positive e negative; nella pratica la prevalenza di superfici sane può essere maggiore.
- Tre seed permettono di stimare in modo preliminare la variabilità del training, ma non forniscono una stima molto precisa della distribuzione dei risultati.
- Il bootstrap appaiato quantifica l'incertezza legata al campione di test, non sostituisce la ripetizione degli addestramenti.
- Le metriche sui difetti piccoli sono informative, ma i sottogruppi possono essere limitati e non tutte le fasce hanno intervalli di confidenza dedicati.
- Le misure di efficienza sono specifiche dell'hardware e del protocollo di benchmark utilizzati.
- Il modello personalizzato è ispirato al paper di riferimento, non una riproduzione integrale di tutti i suoi moduli architetturali.

Per mantenere la riproducibilità, annotare la versione di Python, PyTorch, CUDA, Ultralytics, il modello di GPU, i seed, le configurazioni selezionate e i percorsi dei dati. La cache immagini di YOLO è impostata su `disk` nello script di orchestrazione.

## Riferimenti

- Qing-Yu Xu, Xing Jin, **“MS-YOLOv8-Seg: A multi-stage segmentation framework for industrial data perception”**, *Proceedings of the 2026 6th International Conference on Automation Control, Algorithm and Intelligent Bionics (ACAIB '26)*, pp. 514–521, 2026. DOI: [10.1145/3828836.3828916](https://doi.org/10.1145/3828836.3828916).
- Ultralytics, documentazione YOLO: <https://docs.ultralytics.com/>.
- GhostNet: Kai Han et al., “GhostNet: More Features from Cheap Operations”, CVPR 2020.
- RepVGG: Xiaohan Ding et al., “RepVGG: Making VGG-Style ConvNets Great Again”, CVPR 2021.
- SimAM: Lingxiao Yang, Ru-Yuan Zhang, Lida Li e Xiaohua Xie, “SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks”, Proceedings of the 38th International Conference on Machine Learning (ICML), PMLR 139, pp. 11863–11874, 2021. <https://proceedings.mlr.press/v139/yang21o.html>

---

**Nota:** questo README descrive l'esperimento S2DS e i risultati registrati nel progetto. Per ripetere esattamente le run servono anche il codice sorgente effettivo, il dataset autorizzato, i pesi/checkpoint e l'ambiente software compatibile.
