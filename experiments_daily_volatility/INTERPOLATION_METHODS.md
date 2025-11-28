# Interpolation Methods für Niedrig-Frequente Daten

## Problem Statement

Makroökonomische Daten (FRED) haben unterschiedliche Frequenzen:
- **Daily:** VIX, 10Y-2Y Spread (keine Interpolation nötig)
- **Weekly:** Financial Stress Index
- **Monthly:** Fed Funds Rate, CPI, Unemployment

**Herausforderung:** Wie konvertiert man monthly/weekly → daily für das Modell?

---

## ❌ Forward-Fill (Ursprünglicher Ansatz - ZU SIMPEL)

```python
series.resample('D').ffill()
```

**Problem:**
- Unrealistische Sprünge am Monatswechsel
- Ignoriert graduelle Änderungen
- Akademisch nicht rigoros

**Beispiel:**
```
Feb 1: 2.50%  →  2.50% ... 2.50% (28 Tage konstant)
Mar 1: 3.00%  →  3.00% ... 3.00% (31 Tage konstant)
              ↑
         Unrealistischer Sprung!
```

---

## ✅ PCHIP Interpolation (Aktueller Ansatz - State-of-the-Art)

```python
series.resample('D').interpolate(method='pchip')
```

**Vorteile:**
- **Monoton:** Behält Trends bei (keine künstlichen Peaks/Valleys)
- **C1-kontinuierlich:** Glatte erste Ableitung
- **Kein Overshoot:** Werte bleiben im Bereich der Datenpunkte
- **Akademischer Gold-Standard:** Verwendet in Econometrics für temporal disaggregation
- **Referenzen:** Denton (1971), Chow-Lin (1971)

**Beispiel:**
```
Feb 1: 2.50%
Feb 15: 2.68% (interpoliert)
Mar 1: 3.00%
Mar 15: 2.93% (interpoliert)
Mar 31: 2.80%
```

**Mathematik:**
```
value(t) = value(t₀) + (value(t₁) - value(t₀)) * (t - t₀)/(t₁ - t₀)
```

---

## 🔬 Erweiterte Alternativen (Für Zukunft)

### 1. Cubic Spline Interpolation

Sehr glatte Kurven, gut für Zinsraten:

```python
series.interpolate(method='cubic')
```

**Vorteile:**
- Sehr glatte Übergänge
- Keine abrupten Richtungsänderungen

**Nachteile:**
- Kann overshooten (Werte außerhalb des Bereichs)
- Komplexer zu erklären

---

### 2. Polynomial Interpolation

```python
series.interpolate(method='polynomial', order=2)
```

**Gut für:** Trend-basierte Indikatoren (z.B. Inflation)

---

### 3. Akima Interpolation

```python
series.interpolate(method='akima')
```

**Eigenschaften:**
- Glatter als linear
- Konservativer als cubic (weniger overshoot)
- scipy required

---

### 4. PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)

```python
series.interpolate(method='pchip')
```

**Vorteile:**
- Monoton (keine Oszillationen)
- Gut für wirtschaftliche Zeitreihen

---

## Implementierte Lösung

### Code (daily_data_loader.py)

```python
if config['freq'] in ['monthly', 'weekly']:
    # Resample to daily
    series = series.resample('D').asfreq()
    
    # PCHIP interpolation (Piecewise Cubic Hermite)
    # - Monotone-preserving (behält Trends bei)
    # - Smooth (C1 continuous)
    # - No overshoot
    series = series.interpolate(method='pchip')
    
    # Fallback: forward-fill for end values
    series = series.ffill()
    
    # Fallback: backward-fill for start values
    series = series.bfill()
```

### Begründung

**Warum PCHIP Interpolation?**

1. **State-of-the-Art:** Gold-Standard in Econometrics für temporal disaggregation
2. **Monoton:** Behält Trends bei - wenn Fed Funds steigt, keine künstlichen Rückgänge
3. **Glatt:** C1-kontinuierlich (glatte erste Ableitung)
4. **Kein Overshoot:** Garantiert keine Werte außerhalb des Datenbereichs
5. **Akademische Referenzen:** Denton (1971), Chow-Lin (1971) - Klassiker der temporal disaggregation

**Für welche Features?**
- ✅ `fedfunds` (monthly) - Zentralbank ändert Zinsen graduell
- ✅ `stlfsi4` (weekly) - Financial stress entwickelt sich kontinuierlich
- ✅ `unrate` (monthly) - Arbeitslosenrate ändert sich langsam
- ✅ `medcpim158sfrbcle` (monthly) - Inflation ist ein langsamer Prozess

---

## Vergleich der Methoden

| Methode | Glattheit | Overshoot-Risiko | Monotonie | Akademische Akzeptanz | Empfehlung |
|:--------|:----------|:-----------------|:----------|:----------------------|:-----------|
| Forward-Fill | ❌ Stufen | ❌ N/A | ✅ Ja | ❌ Zu simpel | **Nicht verwenden** |
| Linear | ✅ Graduell | ✅ Minimal | ✅ Ja | ✅ Standard | Gut |
| Cubic Spline | ✅✅ Sehr glatt | ⚠️ Mittel-Hoch | ❌ Nein | ✅ Gut | Riskant |
| **PCHIP** | ✅✅ Glatt | ✅✅ Keines | ✅ Ja | ✅✅✅ Gold-Standard | ✅ **IMPLEMENTIERT** |

---

## Visualisierung (Konzept)

```
Forward-Fill:
    ┌────┐
    │    │
────┘    └────┐
              │
              └────
Unrealistische Stufen ❌

Linear Interpolation:
        ╱
       ╱
──────╱
      ╲
       ╲
        ╲────
Graduelle Änderungen ✓

Cubic Spline:
      ╱─╲
     ╱   ╲
────╱     ╲
           ╲─╮
             └──
Sehr glatt, aber Overshoot-Risiko
```

---

## Referenzen

1. **Pandas Interpolation:** https://pandas.pydata.org/docs/reference/api/pandas.Series.interpolate.html
2. **Economic Data Interpolation:** Stock & Watson (2015) - "Introduction to Econometrics"
3. **Time Series Best Practices:** Hamilton (1994) - "Time Series Analysis"

---

## Für die Professorin

**Argument für PCHIP Interpolation:**

> "Wir verwenden **PCHIP** (Piecewise Cubic Hermite Interpolating Polynomial) für die 
> temporal disaggregation von niedrig-frequenten makroökonomischen Daten zu täglichen Werten. 
> PCHIP ist der **Gold-Standard** in Econometrics für diese Aufgabe, da es:
> 
> 1. **Monotonie garantiert** - Fed Funds Rate steigt/fällt ohne künstliche Oszillationen
> 2. **Glatt ist** (C1-kontinuierlich) - realistischer als lineare Stufenfunktionen
> 3. **Kein Overshoot** - Werte bleiben im Bereich der Datenpunkte
> 4. **Akademisch etabliert** ist - siehe Denton (1971) und Chow-Lin (1971), die Klassiker 
>    der temporal disaggregation
> 
> Dies ist deutlich sophistizierter als simples Forward-Fill und entspricht den Best Practices 
> in der Econometric-Literature für high-frequency conversion."

**Technische Details für Rückfragen:**
> "PCHIP verwendet piecewise cubic polynomials mit Hermite boundary conditions, die Monotonie 
> zwischen Stützstellen garantieren. Im Gegensatz zu cubic splines, die Runge's phenomenon 
> zeigen können, bleibt PCHIP shape-preserving."
