# Kalibrácia kamery a detekcia geometrických tvarov pomocou OpenCV

## 1. Úvod

Cieľom tejto úlohy bolo implementovať kalibráciu kamery, detekciu geometrických tvarov a farebný filter pomocou knižnice OpenCV. Program spracováva obraz z kamery v reálnom čase a dokáže identifikovať základné geometrické objekty v obraze. Súčasťou riešenia je aj odstránenie optického skreslenia kamery a výpočet veľkosti objektov na základe kalibračných parametrov.

Program bol implementovaný v jazyku Python a využíva knižnice:

- OpenCV
- NumPy
- Ximea API (xiapi)

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

Po kalibrácii je možné odstrániť optické skreslenie objektívu. Na tento účel bola použitá funkcia:

```python
cv.undistort()
```

Program načíta uložené parametre kamery zo súboru JSON a aplikuje ich na vstupný obraz. Výsledkom je obraz bez radiálneho a tangenciálneho skreslenia.

> **Poznámka:** Obrázky z pôvodného PDF neboli do tejto Markdown verzie konvertované.
>
> - Figure 1: Pôvodny obraz
> - Figure 2: Korigovany obraz

## 4. Detekcia geometrických tvarov

Detekcia geometrických tvarov bola implementovaná pomocou spracovania obrazu v OpenCV.

Najprv je obraz konvertovaný do grayscalu:

```python
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
```

Pre redukciu šumu je použitý Gaussov filter:

```python
blurred = cv.GaussianBlur(gray, (9, 9), 0)
```

Následne sa vykoná detekcia hrán pomocou Cannyho algoritmu:

```python
edges = cv.Canny(blurred, 20, 60)
```

Kontúry objektov sú získané pomocou:

```python
cv.findContours()
```

Kontúry sú aproximované na polygóny pomocou:

```python
cv.approxPolyDP()
```

Na základe počtu vrcholov je určený typ geometrického objektu:

- 3 vrcholy – trojuholník
- 4 vrcholy – štvorec alebo obdĺžnik
- 5 vrcholov – päťuholník

Kružnice sú detegované pomocou Houghovej transformácie:

```python
cv.HoughCircles()
```

Po detekcii je každý objekt graficky označený a jeho stred je vypočítaný pomocou momentov:

```python
M = cv.moments(contour)
```

> **Poznámka:** Obrázok *Figure 3: Detekcia geometrických tvarov* nebol do tejto Markdown verzie konvertovaný.

## 5. Farebný filter

Farebný filter bol implementovaný v HSV farebnom priestore. Tento farebný model umožňuje jednoduchšie definovať rozsah farieb.

Najprv sa obraz konvertuje do HSV priestoru:

```python
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
```

Používateľ môže meniť rozsah detegovanej farby pomocou trackbarov (grafických posuvníkov). Program následne vytvorí masku pomocou funkcie:

```python
cv.inRange()
```

Pixely patriace do definovaného rozsahu farieb sú následne nahradené inou farbou.

> **Poznámka:** Obrázok *Figure 4: Ukážka farebného filtra* nebol do tejto Markdown verzie konvertovaný.

## 6. Určenie veľkosti objektov

Na výpočet veľkosti objektov boli využité výsledky kalibrácie kamery. Pred meraním je obraz najprv korigovaný pomocou odstránenia skreslenia.

Rozmery objektu v pixeloch sú získané pomocou minimálneho ohraničujúceho obdĺžnika:

```python
rect = cv.minAreaRect(contour)
```

Rozmery sú následne prepočítané na centimetre pomocou ohniskovej vzdialenosti kamery:

```text
width_cm  = (width_px  * distance) / fx
height_cm = (height_px * distance) / fy
```

kde:

- `width_px`, `height_px` sú rozmery objektu v pixeloch
- `distance` je vzdialenosť objektu od kamery
- `fx`, `fy` sú ohniskové vzdialenosti z kalibrácie kamery

Výsledné rozmery objektu sú zobrazené priamo v obraze.

## 7. Záver

V rámci projektu bola úspešne implementovaná kalibrácia kamery, odstránenie optického skreslenia, detekcia geometrických tvarov a farebný filter. Program dokáže spracovávať obraz v reálnom čase a identifikovať základné objekty v obraze. Získané kalibračné parametre umožňujú presnejšie spracovanie obrazu a výpočet reálnych rozmerov objektov.
