# Kalibrácia kamery a detekcia geometrických tvarov pomocou OpenCV

## 1. Úvod

Cieľom tejto úlohy bolo implementovať kalibráciu kamery, detekciu geometrických tvarov a farebný filter pomocou knižnice OpenCV. Program spracováva obraz z kamery v reálnom čase a dokáže identifikovať základné geometrické objekty v obraze. Súčasťou riešenia je aj odstránenie optického skreslenia kamery a výpočet veľkosti objektov na základe kalibračných parametrov.

Program bol implementovaný v jazyku Python a využíva knižnice:

- OpenCV
- NumPy
- Ximea API (xiapi)

## 1.1 Štruktúra projektu

- `task_1/calibration.py` – kalibrácia kamery zo snímok šachovnice
- `task_1/undistort_image.py` – korekcia skreslenia uloženého obrázka pomocou JSON parametrov
- `task_2/shape_detector.py` – detekcia geometrických tvarov, farebná zmena a živé ovládanie pomocou trackbarov

## 2. Kalibrácia kamery

Kalibrácia kamery bola realizovaná pomocou šachovnice (*chessboard pattern*). Z viacerých snímok šachovnice boli detegované rohy pomocou funkcie:

```python
cv.findChessboardCorners()
```

Po detekcii boli rohy spresnené pomocou:

```python
cv.cornerSubPix()
```

Na základe získaných 2D bodov v obraze a známych 3D bodov šachovnice boli vypočítané parametre kamery pomocou funkcie:

```python
cv.calibrateCamera()
```

Výsledkom kalibrácie sú:

- matica vnútorných parametrov kamery
- koeficienty skreslenia objektívu

Tieto parametre sú uložené do súboru JSON pre ďalšie použitie.

### 2.1 Matica kamery

Matica vnútorných parametrov kamery má tvar:

```text
[ fx   0  cx ]
[  0  fy  cy ]
[  0   0   1 ]
```

kde:

- `fx`, `fy` – ohniskové vzdialenosti v pixeloch
- `cx`, `cy` – hlavný bod projekcie

Vypočítaná matica kamery:

```text
[ 3728.72     0    1270.76 ]
[    0     3724.52   994.10 ]
[    0        0        1    ]
```

Koeficienty skreslenia:

```text
dist = [-0.3547, -0.6906, 0.00018, -0.00220, 5.5481]
```

## 3. Odstránenie skreslenia obrazu

Po kalibrácii je možné odstrániť optické skreslenie objektívu pomocou uložených parametrov kamery.  
Skript `task_1/undistort_image.py` načíta maticu kamery a koeficienty skreslenia zo súboru JSON a následne použije:

```python
cv.getOptimalNewCameraMatrix()
cv.undistort()
```

Voliteľne sa výsledný obraz oreže na platnú ROI oblasť. Orezanie je možné vypnúť prepínačom `--no-crop`.

> **Poznámka:** Obrázky z pôvodného PDF neboli do tejto Markdown verzie konvertované.
>
> - Figure 1: Pôvodny obraz
> - Figure 2: Korigovany obraz

## 4. Detekcia geometrických tvarov

Detekcia geometrických tvarov je realizovaná nad grayscale obrazom. Najprv sa obraz skonvertuje a vyhladí pomocou Gaussovho filtra:

```python
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray, (9, 9), 0)
```

Hrany sú následne detegované pomocou Cannyho algoritmu:

```python
thresh_image = cv.Canny(blurred, canny_t1, canny_t2)
```

Hodnoty `canny_t1` a `canny_t2` sú nastaviteľné v reálnom čase pomocou trackbarov v okne **Canny Controls**.

Kontúry objektov sú získané pomocou:

```python
cv.findContours()
```

Aproximácia kontúr prebieha cez:

```python
cv.approxPolyDP()
```

Typ objektu sa určuje podľa počtu vrcholov aproximovaného polygónu:

- 3 vrcholy – trojuholník
- 4 vrcholy – štvorec alebo obdĺžnik
- 5 vrcholov – päťuholník

Kružnice sú detegované samostatne pomocou Houghovej transformácie:

```python
cv.HoughCircles()
```

Po detekcii je každý objekt graficky označený a jeho stred je vypočítaný pomocou momentov:

```python
M = cv.moments(contour)
```

> **Poznámka:** Obrázok *Figure 3: Detekcia geometrických tvarov* nebol do tejto Markdown verzie konvertovaný.

## 5. Farebný filter a zmena farby

Program obsahuje interaktívnu zmenu farby objektov. Používateľ nastavuje:

- pôvodnú farbu (**From H, From S, From V**)
- cieľovú farbu (**To H, To S, To V**)

Tieto hodnoty sa menia pomocou trackbarov v okne **HSV Controls**.

Implementácia pracuje tak, že RGB obraz skonvertuje do HSV priestoru, vytvorí masku na základe tolerancií farby a následne nahradí vybrané pixely novou farbou.

Použité sú najmä funkcie:

```python
cv.cvtColor()
cv.inRange()
```

Voliteľne je možné zachovať zložky saturácie a jasu a meniť iba odtieň farby.

## 6. Určenie veľkosti objektov

Výpočet veľkosti objektov využíva kalibračné parametre kamery a známu vzdialenosť objektu od kamery. Pred samotným meraním je obraz korigovaný od skreslenia.

Rozmery objektu v pixeloch sú získané pomocou:

```python
rect = cv.minAreaRect(contour)
```

Následne sa prepočítajú na centimetre:

```text
width_cm  = (width_px  * distance_cm) / fx
height_cm = (height_px * distance_cm) / fy
```

kde:

- `width_px`, `height_px` sú rozmery objektu v pixeloch
- `distance_cm` je vzdialenosť objektu od kamery
- `fx`, `fy` sú ohniskové vzdialenosti z matice kamery

V aktuálnej implementácii je meranie veľkosti povolené pri spracovaní obrazu z kamery Ximea. Pri webkamere je detekcia spustená bez merania rozmerov.

### 6.1 Konfiguračné parametre

Skript `shape_detector.py` používa konfiguračný slovník, v ktorom sa nastavujú napríklad:

- `old_color`, `new_color`
- `canny_threshold1`, `canny_threshold2`
- `measure_size`
- `distance_cm`
- `camera_matrix`
- `dist_coeffs`
- `ximea_display_scale`

Kalibračné parametre sú v aktuálnej verzii vložené priamo v kóde. Je možné ich nahradiť načítaním zo súboru `camera_params.json`.

## 7. Záver

V rámci projektu bola úspešne implementovaná kalibrácia kamery, odstránenie optického skreslenia, detekcia geometrických tvarov a farebný filter. Program dokáže spracovávať obraz v reálnom čase a identifikovať základné objekty v obraze. Získané kalibračné parametre umožňujú presnejšie spracovanie obrazu a výpočet reálnych rozmerov objektov.

## 3.1 Spustenie skriptov

### Kalibrácia kamery

Zo zložky `task_1`:

```powershell
python calibration.py
```

Skript načíta obrázky zo zložky `./images/` a uloží výsledok do `camera_params.json`.

### Korekcia skreslenia obrázka

```powershell
python undistort_image.py image.jpg
```

Použitie vlastného JSON súboru a výstupného názvu:

```powershell
python undistort_image.py image.jpg --params camera_params.json --output result.jpg
```

Bez orezania výsledku:

```powershell
python undistort_image.py image.jpg --no-crop
```

### Detekcia tvarov

Zo zložky `task_2`:

```powershell
python shape_detector.py
```
