## Použité prostredie a dáta

Použité bolo pripravené pracovné prostredie s nástrojmi COLMAP, 3D Gaussian Splatting a SIBR Gaussian Viewer. Dáta tvorili fotografie vlastnej interiérovej scény nasnímanej v bloku D na 5. poschodí.

**Parametre datasetu, kamery a fotografií**

| Parameter | Hodnota |
|---|---|
| Dataset | vlastná interiérová scéna |
| Miesto snímania | blok D, 5. poschodie |
| Snímaná scéna | Rubikova kocka, dve hokejky, tenisová raketa s loptičkou |
| Použité zariadenie | Apple iPhone 13 |
| Typ kamery | širokouhlá kamera |
| Ohnisková vzdialenosť | 26 mm |
| Clona | f/1.6 |
| Expozičný čas | 1/60 s |
| ISO | 200 |
| Kompenzácia expozície | 0 EV |
| Rozlíšenie snímača | 12 Mpx |
| Počet nasnímaných fotografií | 174 |
| Počet fotografií použitých na spracovanie | 151 |
| Pôvodné rozlíšenie fotografií | 4032x3024 |
| Rozlíšenie po downscale | 1600x1200 |
| Vstupný formát fotografií | HEIF |
| Formát po príprave dát | JPG |
| Spôsob snímania | postupný pohyb okolo scény |
| Svetelné podmienky | interiérové osvetlenie |
| Nastavenia kamery | automatické nastavenia telefónu |

## Postup spracovania

Pre vlastnú scénu bol použitý nasledujúci postup:

1. Spustenie pracovného prostredia.
2. Príprava fotografií: výber použiteľných záberov, konverzia z HEIF do JPG a zmenšenie rozlíšenia.
3. Spustenie `colmap gui`.
4. Vytvorenie COLMAP projektu.
5. Feature extraction s modelom `SIMPLE_PINHOLE`.
6. Feature matching v režime *Exhaustive*.
7. Sparse reconstruction.
8. Export modelu do `distorted/sparse/0/`.
9. Spustenie `convert.py` na undistortion fotografií.
10. Trénovanie 3DGS modelu do 30 000 iterácií.
11. Vizualizácia výsledku cez `SIBR_gaussianViewer_app`.

```bash
./run.sh
colmap gui
mkdir -p /workspace/data/<dataset>/sparse
mkdir -p /workspace/data/<dataset>/distorted/sparse/0
python3 convert.py -s data/<dataset> --skip_matching
python3 train.py -s data/<dataset> --data_device cpu
SIBR_gaussianViewer_app -m output/<uuid>
```

Pri nedostatku pamäte bol použitý parameter na zníženie rozlíšenia:

```bash
python3 train.py -s data/<dataset> --data_device cpu --resolution 2
```

## Vlastná scéna nasnímaná pomocou Apple iPhone 13

Vlastnú scénu sme spolu s kolegom nasnímali pomocou Apple iPhone 13 v bloku D na 5. poschodí. Scéna obsahovala Rubikovu kocku, dve hokejky a tenisovú raketu s loptičkou. Fotografie boli urobené postupným pohybom okolo scény tak, aby medzi susednými zábermi vznikalo dostatočné prekrytie. Snažili sme sa udržať scénu statickú a snímať ju z viacerých uhlov pohľadu.

Celkovo bolo nasnímaných 174 fotografií, z ktorých bolo po výbere použitých 151. Vstupné fotografie boli vo formáte HEIF s pôvodným rozlíšením 4032x3024 px. Pred spustením COLMAP boli skonvertované do formátu JPG a zmenšené na 1600x1200 px so zachovaním pomeru strán. Tým sa znížila výpočtová náročnosť feature extraction, matching a následného trénovania 3D Gaussian Splatting modelu.

![Ukážka fotografie po downscale.](IMG_4221.jpg)

![Feature extraction pre vlastnú scénu.](bobik.jpg)

![Výsledok vlastnej scény v SIBR Gaussian Viewer.](render.jpg)

**PSNR hodnoty pre vlastnú scénu**

| Iterácia | PSNR [dB] |
|---|---|
| 7 000 | neuvedené |
| 30 000 | neuvedené |

## Vyhodnotenie výsledkov

### Vizuálne vyhodnotenie

Výsledný model bol skontrolovaný v SIBR Gaussian Viewer-i. Hodnotená bola najmä čitateľnosť scény, ostrosť detailov, stabilita geometrie a výskyt artefaktov. Vlastná scéna bola náročnejšia na spracovanie, pretože fotografie vznikli v bežných interiérových podmienkach pomocou mobilného telefónu.

![Render vlastnej scény z prvého uhla pohľadu.](render2.jpg)

![Render vlastnej scény z druhého uhla pohľadu.](render3.jpg)

### Faktory ovplyvňujúce kvalitu vlastnej scény

Kvalitu vlastnej rekonštrukcie ovplyvnili najmä:

- počet fotografií a ich prekrytie,
- rovnomernosť osvetlenia,
- typ objektov a povrchov v scéne,
- ostrosť fotografií,
- rozlíšenie po downscale,
- automatické nastavenia kamery v telefóne,
- konverzia fotografií z HEIF do JPG,
- počet úspešne zrekonštruovaných kamier v COLMAP.

Pri snímaní pomocou iPhone 13 mohli kvalitu ovplyvniť automatické nastavenia telefónu, napríklad automatická expozícia, zaostrovanie alebo interné spracovanie obrazu. Aj keď EXIF parametre ukazujú rovnaké základné nastavenia snímania, telefón môže medzi zábermi stále meniť lokálne spracovanie obrazu, vyváženie bielej, ostrenie alebo redukciu šumu. To môže znížiť konzistenciu vstupných dát pre COLMAP aj pre následné trénovanie 3DGS.

Použitá clona f/1.6 prepúšťa veľa svetla, ale zároveň môže znížiť hĺbku ostrosti. Pri snímaní objektov v rôznych vzdialenostiach preto mohli byť niektoré časti scény menej ostré. Expozičný čas 1/60 s je v interiéri použiteľný, ale pri pohybe rukou môže spôsobiť mierne rozmazanie niektorých fotografií. ISO 200 je relatívne nízke, takže šum by nemal byť zásadný problém, no interiérové osvetlenie aj tak mohlo vytvoriť tiene a nerovnomerné nasvietenie.

Ďalším možným problémom bola samotná kompozícia scény. Rubikova kocka má dobrú textúru a výrazné hrany, čo pomáha feature extraction. Hokejky, raketa a loptička však obsahujú tenšie štruktúry a povrchy, ktoré môžu byť pre rekonštrukciu náročnejšie. Ak sa v scéne nachádzali lesklé časti, odlesky sa mohli meniť podľa uhla pohľadu a tým zhoršiť matching medzi fotografiami.

Napriek týmto obmedzeniam výsledný model zachytáva základnú geometriu vlastnej scény a umožňuje jej interaktívne zobrazenie vo viewer-i.

## Záver a odporúčania

Postup 3D Gaussian Splatting bol vykonaný na vlastnej scéne nasnímanej pomocou Apple iPhone 13. V práci bol použitý celý pipeline: príprava fotografií, COLMAP rekonštrukcia, export modelu, undistortion, trénovanie 3DGS modelu a vizualizácia vo viewer-i.

Výsledok ukázal, že aj fotografie z mobilného telefónu môžu byť použité na vytvorenie interaktívnej 3D reprezentácie scény. Kvalita výsledku však závisí od počtu fotografií, prekrytia medzi zábermi, osvetlenia, ostrosti a konzistentnosti nastavení kamery.

Pre lepší výsledok by sme nabudúce:

- nasnímali viac fotografií z viacerých výšok,
- použili rovnomernejšie difúzne osvetlenie,
- vybrali scénu s výraznejšou textúrou a menším množstvom lesklých povrchov,
- udržali čo najkonzistentnejšie nastavenia kamery,
- priebežne kontrolovali ostrosť snímok,
