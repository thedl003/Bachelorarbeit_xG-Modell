# Bachelorarbeit: xG-Modell

Python-Pipeline zur Entwicklung und Evaluation eines Expected-Goals-Modells (xG) auf Basis frei zugänglicher StatsBomb-Event-Daten der Fußball-Weltmeisterschaft 2022. Teil meiner Bachelorarbeit im Studiengang B.Sc. Informatik an der IU Internationale Hochschule.

## Über das Projekt

Ziel der Arbeit ist es, mit möglichst einfachen, transparenten Mitteln (logistische Regression, ausschließlich frei verfügbare Event-Daten) ein xG-Modell zu entwickeln und dessen Vorhersagekraft mit dem kommerziellen xG-Modell von StatsBomb zu vergleichen. Untersucht werden drei Modellvarianten:

1. **Baseline-Heuristik** – einfache Distanzregel (Schuss innerhalb von 16.5 m = 15 % Torwahrscheinlichkeit, sonst 1 %)
2. **LR (Geometrie)** – logistische Regression ausschließlich mit den geometrischen Merkmalen Distanz und Schusswinkel
3. **LR (Erweitert)** – logistische Regression zusätzlich mit kategorialem Spielkontext (Körperteil, Spielsituation)

Alle drei Modelle werden abschließend gegen das offizielle StatsBomb-xG-Modell gebenchmarkt.

## Datengrundlage

Frei zugängliche Event-Daten von StatsBomb (WM 2022, Männer) über die offizielle Python-Bibliothek [statsbombpy](https://github.com/statsbomb/statsbombpy). Nach Filterung (Ausschluss von Elfmetern, Entfernung fehlender Werte) umfasst der finale Datensatz 1430 Torschüsse.

## Methodik

- **Feature Engineering:** Berechnung von Distanz und sichtbarem Schusswinkel aus den X/Y-Koordinaten
- **Kodierung:** One-Hot-Encoding kategorialer Merkmale (Körperteil, Spielsituation)
- **Training:** 80/20 Train-Test-Split mit Stratifizierung, logistische Regression (scikit-learn)
- **Evaluation:** Log-Loss, ROC-AUC und Brier Score 

## Ergebnisse

| Modell | Log-Loss ↓ | ROC-AUC ↑ | Brier Score ↓ |
|---|---|---|---|
| Baseline-Heuristik | 0.3165 | 0.7105 | 0.0889 |
| LR (Geometrie) | 0.2831 | 0.7741 | 0.0817 |
| LR (Erweitert) | 0.2670 | 0.8169 | 0.0771 |
| StatsBomb (Benchmark) | 0.2595 | 0.8285 | 0.0748 |

Das erweiterte Modell erreicht mit einfachen, frei verfügbaren Event-Daten eine Vorhersagequalität nahe am kommerziellen StatsBomb-Benchmark.

## Installation

```bash
pip install -r requirements.txt
```

## Nutzung

```bash
python xg_model_base.py
```

Beim ersten Ausführen werden die Daten über die StatsBomb-API geladen und lokal als `shots_wm2022.csv` zwischengespeichert (spätere Läufe nutzen den Cache). Die Konsolenausgabe enthält Datenprofiling, Feature-Importance, drei Fallstudien sowie die finale Evaluationstabelle. Zusätzlich werden `shotmap_export.csv` und `roc_curve_export.csv` für weiterführende Visualisierungen exportiert.

## Projektstruktur
