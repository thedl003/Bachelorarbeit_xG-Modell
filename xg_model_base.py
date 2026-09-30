import math
import os
import pandas as pd
import numpy as np
from statsbombpy import sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

# Konstanten für die Geometrie des Spielfelds (StatsBomb-Koordinatensystem)
GOAL_X = 120.0
GOAL_POST_Y1 = 36.0 
GOAL_POST_Y2 = 44.0
GOAL_CENTER_Y = 40.0

def calculate_distance(x, y):
    """
    Berechnen der Distanz (Luftlinie) vom Schusspunkt (x, y) 
    zur Mitte des Tores (GOAL_X, GOAL_CENTER_Y).
    """
    x_diff = GOAL_X - x
    y_diff = GOAL_CENTER_Y - y
    return math.sqrt(x_diff**2 + y_diff**2)

def calculate_angle(x, y):
    """
    Berechnen des Winkels des Schusspunktes zur Tormitte [Grad]
    mithilfe des Kosinussatzes im Dreieck (Schusspunkt - Pfosten 1 - Pfosten 2).
    """
    a = math.sqrt((GOAL_X - x)**2 + (GOAL_POST_Y1 - y)**2) # Berechnen der Luftlinie zu Pfosten 1
    b = math.sqrt((GOAL_X - x)**2 + (GOAL_POST_Y2 - y)**2) # Berechnen der Luftlinie zu Pfosten 2
    c = abs(GOAL_POST_Y2 - GOAL_POST_Y1) #Wir nutzen abs um den Betrag zu erhalten
    
    if a == 0 or b == 0: #Dann wissen wir dass wir auf der Grundlinie sind 
        return 0
        
    try:
        cos_alpha = (a**2 + b**2 - c**2) / (2 * a * b) # Nutzen des Kosinussatzes
        cos_alpha = max(-1.0, min(1.0, cos_alpha)) # Sicherstellen, dass durch Rundung kein Fehler entsteht
        return math.degrees(math.acos(cos_alpha))
    except Exception:
        return 0.0

def fetch_season_shots(competition_id=43, season_id=106): # Das sind die internen IDs für die WM 2022 der Männer
    """
    Lädt alle Spiele einer Saison (Standard: WM 2022) herunter,
    filtert Schüsse und bereinigt die Daten.
    """
    print(f"Lade Spielplan für Wettbewerb {competition_id}, Saison {season_id}...")
    matches = sb.matches(competition_id=competition_id, season_id=season_id) # Nutzen des API - Aufrufs um einen Pandas DF zu bekommen 
    match_ids = matches['match_id'].tolist() # Extrahieren der Match_IDs aus dem Panda DF 
    print(f"Gefundene Spiele: {len(match_ids)}. Starte Download...")

    all_shots = [] # Arbeiten mit einer List um Performance zu sichern (DF wäre in einer Schleife zu langsam)
    # total_events_count = 0
    # with_penalty = 0
    for id, m_id in enumerate(match_ids, 1): # Nutzen von Enumerate um Zählervariable zu vermeiden und um bei 1 zu starten
        try:
            events = sb.events(match_id=m_id) # Laden aller Events in einen DF
            # total_events_count += len(events)
            if 'type' not in events.columns or 'Shot' not in events['type'].unique(): # Überspringen des Matches wenn es keinen Schuss gab
                continue

            # Data Filter 1: Behalten von Schüssen
            shots_df = events[events['type'] == 'Shot'].copy() # Kopieren von Events die vom Typ "Shot" sind 
            # with_penalty += len(shots_df)
            # Data Filter 2: Herausfiltern von Elfmetern
            if 'shot_type' in shots_df.columns: # Aus unserer neuen Tabelle den Typen des "Shot" extrahieren
                shots_df = shots_df[shots_df['shot_type'] != 'Penalty'] # Behalten von Schüssen die kein Elfmeter sind

            # Iteriere durch jeden Schuss und extrahiere Features
            for i, row in shots_df.iterrows(): # Durchlaufen jeder Zeile mithilfe von iterrow
                location = row.get('location')
                if not isinstance(location, list) or len(location) < 2: # Überspringen des Schusses, wenn es nur eine oder keine Koordinate gibt
                    continue

                x = location[0] # Speichern der Koordinaten
                y = location[1]
                dist = calculate_distance(x, y) # Berechnen von Distanz und Winkel
                angle = calculate_angle(x, y)

                # Data Filter 3: Binäres Label (1 für Tor, 0 für alles andere)
                outcome = str(row.get('shot_outcome', '')).strip() # Erstellen eines Strings "Outcome" mit "Goal" oder nicht
                if outcome == 'Goal': 
                    is_goal = 1  
                else: 
                    is_goal = 0

                #Speichern von verschiedenen Metadaten zu dem Schuss
                body_part = str(row.get('shot_body_part', 'Unknown')) 
                play_pattern = str(row.get('play_pattern', 'Unknown'))
                statsbomb_xg = row.get('shot_statsbomb_xg', np.nan)

                all_shots.append({ # Erstellen einer Liste von Dictionaries mit den Metadaten
                    'match_id': m_id,
                    'player': str(row.get('player', 'Unknown')),
                    'x': x,
                    'y': y,
                    'distance_to_goal': dist,
                    'visible_angle': angle,
                    'shot_body_part': body_part,
                    'play_pattern': play_pattern,
                    'is_goal': is_goal,
                    'statsbomb_xg': statsbomb_xg
                })

            if id % 10 == 0 or id == len(match_ids): # Informieren des Users über den Fortschritt im Terminal immer nach 10 Matches oder nach Fertigstellung (verhindern von Spam)
                print(f"Progress: {id}/{len(match_ids)} Spiele verarbeitet...")

        except Exception as e:
            print(f"Fehler bei Spiel {m_id}: {e}")
    # print(f"Gesamtzahl aller Roh-Events: {total_events_count}")
    # print(f"Gesamtanzahl an Schüssen: {with_penalty}")
    df = pd.DataFrame(all_shots) # Erstellen eines DF um sauber weiterarbeiten zu können (Liste nur für Performance Zwecke)
    print(f"\nErfolgreich extrahiert: {len(df)} Schüsse aus {len(match_ids)} Spielen!")
    return df

def verify_data_quality(df):
    """
    Überprüfen der Qualität und Balance der extrahierten Schussdaten .
    """
    print("\n--- DATA PROFILING & QUALITÄTSCHECK ---")
    print(f"Gesamtanzahl Schüsse: {len(df)}")
    print(f"Tore (1): {df['is_goal'].sum()} ({(df['is_goal'].mean()*100):.2f}%)")
    print(f"Fehlschüsse (0): {(df['is_goal'] == 0).sum()} ({((1-df['is_goal'].mean())*100):.2f}%)")
    print(f"{(df['is_goal'].mean()*100):.2f}% der Schüsse waren ein Tor")
    print("Fehlende Werte (NaNs) in primären Features:")
    print(df[['distance_to_goal', 'visible_angle', 'is_goal']].isnull().sum())
    print("---------------------------------------\n")

def run_xg_experiments(df):
    """
   Durchführen des wissenschaftlichen Vergleichsexperiment (Train-Test-Split 80/20):
    1. Baseline-Heuristik (Daumen - Regel: Schüsse in der Box)
    2. Logistische Regression (Geometrische Features)
    3. Logistische Regression (Erweiterte Features)
    4. Vergleich mit StatsBomb Benchmark
    """
    print("--- STARTE MODELLIERUNG & EVALUATION ---")

    # NaN bereinigen
    df_clean = df.dropna(subset=['distance_to_goal', 'visible_angle', 'is_goal']).copy() # Kopieren der Zeilen in der Distanz, Winkel und Torerfolg ungeleich Nan sind

    # Pearson-Korrelation zwischen Distanz und Winkel (vollständiger Datensatz, vor dem Split)
    korrelation = df_clean['distance_to_goal'].corr(df_clean['visible_angle']) # Berechnen der Korrelation
    print(f"\nPearson-Korrelation (distance_to_goal, visible_angle): {korrelation:.4f} (n={len(df_clean)})") # Ausgabe

    # One-Hot-Encoding für kategoriale Features
    body_part_dummies = pd.get_dummies(df_clean['shot_body_part'], prefix='body', drop_first=True) #Nutzen von Dummies für One - Hot - Encoding, mit Prefix "body" für Übersicht, Drop first verhindert redundante Informationen
    play_pattern_dummies = pd.get_dummies(df_clean['play_pattern'], prefix='pattern', drop_first=True)
    
    # Feature-Matrizen definieren
    X_geo = df_clean[['distance_to_goal', 'visible_angle']] # Lernvariable als Konvention immer X
    X_ext = pd.concat([X_geo, body_part_dummies, play_pattern_dummies], axis=1) # Verbinden mithilfe von concat als eine Zeile (axis = 1)
    y = df_clean['is_goal'] # Ergebnis bekommt immer Y

    # Train-Test-Split (80% Training, 20% Evaluation) mit STRATIFIZIERUNG (Gleiche Tor-Quote!)
    X_geo_train, X_geo_test, y_train, y_test = train_test_split(X_geo, y, test_size=0.2, random_state=42, stratify=y) # Nutzen der scikit - Funktion
    X_ext_train, X_ext_test, _, _ = train_test_split(X_ext, y, test_size=0.2, random_state=42, stratify=y) # Nutzen von Wegwerfvariablen, da wir die y - variablen schon haben

    # 1. Baseline-Heuristik (Bauchgefühl-Regel)
    # Strafraumgrenze liegt bei ca. 16.5m Distanz
    # Dumme Regel: Schuss < 16.5m -> 15% Torwahrscheinlichkeit, ansonsten 1%
    baseline_pred = np.where(X_geo_test['distance_to_goal'] <= 16.5, 0.15, 0.01) # Nutzen von WENN DANN SONST von Numpy

    # 2. Modell 1: Logistische Regression (Nur Geometrie)
    model_geo = LogisticRegression() # Erstellen eines leeren Modells
    model_geo.fit(X_geo_train, y_train) # Trainieren des Modells
    pred_geo = model_geo.predict_proba(X_geo_test)[:, 1] # Testen des Modells, Wahrscheinlichkeit für kein Tor wird ignoriert

    # 3. Modell 2: Logistische Regression (Erweitert um Körperteile)
    model_ext = LogisticRegression(max_iter=1000) # Mehr Lernzeit wird benötigt aufgrund der höheren Komplexität
    model_ext.fit(X_ext_train, y_train)

    # Einfluss der Features herausfinden
    features = X_ext_train.columns # Die Spaltennamen der Feature-Matrix um Spaltennamen die Koeffizienten zuzuordnen (selbe Reihenfolge in Ausgabe)
    koeffizienten = model_ext.coef_[0] # Die Koeefizienten des Models sind in 2D Matrix des trainierten Modells, [0] um auf Inhalt zuzugreifen
    odds_ratios = np.exp(koeffizienten) # exp als Umkehrfunktion um die Odds zu erhalten

    importance_df = pd.DataFrame({ # Erstellen eines übersichtlichen Dictionarys aus den 3 Spalten
        'Feature': features,
        'Koeffizient (Log-Odds)': koeffizienten,
        'Odds Ratio (Multiplikator)': odds_ratios
    })
    importance_df = importance_df.sort_values(by='Koeffizient (Log-Odds)', ascending=False) # Absteigende Liste, damit die größten Einflussfaktoren oben stehen

    print("\n--- FEATURE IMPORTANCE (LR Erweitert) ---")
    print(importance_df.to_string(index=False)) # Ignorieren der Indizes des DF

    pred_ext = model_ext.predict_proba(X_ext_test)[:, 1] # Nur Spalte 1 ist relevant - wie wahrscheinlich ist ein Tor

    # Drei exemplarische Schüsse aus dem Testdatensatz extrahieren
    fallstudien_df = df_clean.loc[X_ext_test.index].copy() # Die Herausgefilterten Strings über den Zeilenindex zurückholen und kopieren
    fallstudien_df['xg_erweitert'] = pred_ext # Neue Spalte mit xG-Wert

    shotmap_df = df_clean.loc[X_ext_test.index, ['x', 'y', 'distance_to_goal', 'visible_angle', 'is_goal']].copy()
    shotmap_df['xg_erweitert'] = pred_ext
    shotmap_df.to_csv("shotmap_export.csv", index=False)
    print("Shot-Map-Daten exportiert: shotmap_export.csv")

    # Szenario A: Distanz-Effekt (Schuss aus großer Distanz, > 25m)
    szenario_a = fallstudien_df[fallstudien_df['distance_to_goal'] > 25].sort_values('distance_to_goal', ascending=False).head(1) # Schüsse mit hoher Distanz absteigend sortiert und den mit der größten Distanz wählen

    # Szenario B: Kopfball-Malus (Kopfball im Strafraum, Distanz < 16.5m)
    szenario_b = fallstudien_df[(fallstudien_df['shot_body_part'] == 'Head') & (fallstudien_df['distance_to_goal'] < 16.5)].head(1) # Kopfball und Distanz unter 16.5 meter

    # Szenario C: Konter-Bonus (Schuss mit dem Fuß im Strafraum nach Konter)
    szenario_c = fallstudien_df[
        (fallstudien_df['play_pattern'] == 'From Counter') & # Spielzug ist zwingend ein Konter
        (fallstudien_df['shot_body_part'] != 'Head') & # Zwingend kein Kopfball
        (fallstudien_df['distance_to_goal'] < 16.5) # Abschluss im Strafraum
    ].head(1) # Wähle den Ersten 

    spalten = ['match_id', 'player', 'distance_to_goal', 'visible_angle', 'shot_body_part', 'play_pattern', 'is_goal', 'xg_erweitert']

    print("\n--- SZENARIO A: DISTANZ-EFFEKT ---")
    print(szenario_a[spalten].to_string(index=False))

    print("\n--- SZENARIO B: KOPFBALL-MALUS ---")
    print(szenario_b[spalten].to_string(index=False))

    print("\n--- SZENARIO C: KONTER-BONUS ---")
    print(szenario_c[spalten].to_string(index=False))

    # 4. Benchmark: StatsBomb xG Modell
    missing_xg_count = df_clean.loc[y_test.index, 'statsbomb_xg'].isna().sum() # Überprüfen ob das Ergebnis durch NaN - Werte im sb-modell verfälscht wird 
    missing_xg_pct = (missing_xg_count / len(y_test)) * 100 # Errechnen des relativen Anteils der NaN in Prozent
    print(f"\nHinweis: {missing_xg_count} von {len(y_test)} Test-Schüssen ({missing_xg_pct:.2f}%) haben keinen StatsBomb-xG-Wert und wurden für den Benchmark-Vergleich mit 0.0 aufgefüllt.") # Nutzerausgabe
    statsbomb_test = df_clean.loc[y_test.index, 'statsbomb_xg'].fillna(0.0) # Extrahieren aller statsbomb_xg - Werte der relevanten Zeilen (die Zeilen die wir zum Testen nutzen)


    # Nutzen von scikit um Test und anschließend Vergleich durchzuführen (als Dictionary)
    results = {
        'Modell': ['1. Baseline-Heuristik', '2. LR (Geometrie)', '3. LR (Erweitert)', '4. StatsBomb (Benchmark)'],
        'Log-Loss (niedriger=besser)': [
            log_loss(y_test, baseline_pred), # Log - Loss Vergleich - echtes Ergebnis, vorhergesagtes Ergebnis
            log_loss(y_test, pred_geo),
            log_loss(y_test, pred_ext),
            log_loss(y_test, statsbomb_test)
        ],
        'ROC-AUC (hoeher=besser)': [
            roc_auc_score(y_test, baseline_pred),
            roc_auc_score(y_test, pred_geo),
            roc_auc_score(y_test, pred_ext),
            roc_auc_score(y_test, statsbomb_test)
        ],
        'Brier Score (niedriger=besser)': [
            brier_score_loss(y_test, baseline_pred),
            brier_score_loss(y_test, pred_geo),
            brier_score_loss(y_test, pred_ext),
            brier_score_loss(y_test, statsbomb_test)
        ]
    }

    results_df = pd.DataFrame(results) # Erstellen eines df aus dem Ergebnis - Dictionary
    print("\nEVALUATIONSERGEBNISSE (TEST-DATENSET):")
    print(results_df.to_string(index=False)) # Formatierung des df für die Ausgabe in der Konsole

    from sklearn.metrics import roc_curve

    roc_rows = [] # Liste von Dictionarys erstellen
    for name, scores in [ # Eine Schleife für das Tupel, läuft für jedes Modell einmal durch
        ("Baseline", baseline_pred),
        ("LR Geometrie", pred_geo),
        ("LR Erweitert", pred_ext),
        ("StatsBomb Benchmark", statsbomb_test),
    ]:
        fpr, tpr, _ = roc_curve(y_test, scores) # Das Modell errechnet fals positive und true positive von y_test und scores anhand von Schwellenwerten
        for f, t in zip(fpr, tpr): # Die Zip funktion erstellt die Koordinaten für einen Punkt aus fpr und tpr
            roc_rows.append({"Modell": name, "FPR": f, "TPR": t})

    roc_df = pd.DataFrame(roc_rows) # Umwandeln der Liste in ein Df mit 3 Spalten
    roc_df.to_csv("roc_curve_export.csv", index=False) # Nummerierung der Spalten wird verhindert
    print("ROC-Kurven-Daten exportiert: roc_curve_export.csv") 

    return results_df

if __name__ == "__main__":
    print("\n==========================================")
    print("   xG-MODELL PIPELINE (WM 2022)   ")
    print("==========================================\n")

 # !Wichtig! Sollte calculate_distance/calculate_angle geändert werden, muss auch die csv gelöscht und neu geladen werden
    # Phase 1: Lade komplettes Turnier (WM 2022: competition_id=43, season_id=106)
    if os.path.exists("shots_wm2022.csv"): # Durchsuchen der Festplatte um ständiges herunterladen zu verhindern
        print("Nutze Cache: shots_wm2022.csv")
        dataset = pd.read_csv("shots_wm2022.csv")
    else:
        dataset = fetch_season_shots(competition_id=43, season_id=106) # Sollte die CSV nicht gefunden werden, Daten herunterladen und in CSV schreiben
        dataset.to_csv("shots_wm2022.csv", index=False)

    # Phase 2: Qualitätscheck durchführen 
    verify_data_quality(dataset)

    # Phase 3 & 4: Vergleichsexperiment ausführen und Metriken berechnen
    eval_table = run_xg_experiments(dataset)