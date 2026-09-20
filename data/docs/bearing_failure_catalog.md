# Bearing Failure Catalog — IMS Dataset and ISO 15243 Classification

## Source
- ISO 15243:2017 — Rolling bearings: Damage and failures — Terms, characteristics, causes
- IMS (Intelligent Maintenance Systems) Bearing Dataset documentation
- University of Cincinnati Center for Intelligent Maintenance Systems

---

## 1. IMS Dataset Failure Summary

The IMS dataset contains three accelerated run-to-failure tests conducted at the
University of Cincinnati IMS Center. All tests used identical SKF 6205-2RS deep
groove ball bearings on a shaft rotating at 2000 RPM under 6,000 lbs radial load.

### Test 1 (October–November 2003): 2,156 snapshots, 35 days
- **Duration**: 35 days (842 hours)
- **Failed components**: Bearing 3 and Bearing 4 (both outer races)
- **Failure type**: Outer race spalling — progressive fatigue crack propagation
- **Channels**: 8 (Bearing1_x, Bearing1_y, Bearing2_x, Bearing2_y, Bearing3_x, Bearing3_y, Bearing4_x, Bearing4_y)
- **Early warning indicators**: Kurtosis rise in Bearing 3 channels at ~700 hours

### Test 2 (February 2004): 984 snapshots, 164 hours
- **Duration**: 164 hours (6.8 days) — shortest test
- **Failed component**: Bearing 1
- **Failure type**: Outer race spalling with sudden acceleration in last 8 hours
- **Channels**: 4 (Bearing1, Bearing2, Bearing3, Bearing4)
- **Early warning indicators**: Slight kurtosis increase at ~140 hours, then rapid spike at ~156 hours

### Test 3 (March–April 2004): 6,324 snapshots, 1,053 hours
- **Duration**: 1,053 hours (43.9 days) — longest test
- **Failed component**: Bearing 3
- **Failure type**: Outer race spalling
- **Channels**: 4 (Bearing1, Bearing2, Bearing3, Bearing4)
- **Early warning indicators**: Gradual RMS increase in Bearing 3 from ~800 hours onwards

---

## 2. ISO 15243 Failure Mode Classification

### Category 1: Fatigue (Surface and Subsurface)
**Outer Race Spalling** (most common in IMS dataset)
- Root cause: Hertzian contact stress exceeding fatigue limit; micro-crack initiation at subsurface inclusion
- Visual: Pitting and flaking on outer raceway surface, debris in lubricant
- Vibration signature: Progressive RMS increase, BPFO frequency peak in FFT, high kurtosis
- Progression: Slow (months) then sudden acceleration in final hours

**Inner Race Spalling**
- Root cause: Misalignment causing uneven load distribution; shaft deflection
- Visual: Pitting on inner raceway, often localized at load zone
- Vibration signature: BPFI frequency peak, sidebands at shaft frequency
- Progression: Similar to outer race but modulated by shaft rotation

**Rolling Element (Ball) Fatigue**
- Root cause: Heavy point loading, contamination, electrical discharge
- Visual: Flaking on ball surface, flat spots
- Vibration signature: BSF peak, random high-frequency bursts

---

### Category 2: Wear
**Abrasive Wear**
- Root cause: Contaminated lubricant (particles > 3 microns), inadequate filtering
- Visual: Uniformly worn surfaces, loss of original finish
- Vibration signature: Gradual broadband noise increase, no specific frequency peak
- Prevention: Oil cleanliness ISO 16/14/11 or better

**Adhesive Wear (Smearing)**
- Root cause: Insufficient lubrication, high speed starts, sudden overloads
- Visual: Material transfer from one surface to another, burnished appearance
- Vibration signature: Intermittent high-amplitude events
- Prevention: Adequate viscosity EP (extreme pressure) grease

---

### Category 3: Corrosion
**Moisture Corrosion**
- Root cause: Water ingress (condensation, seal failure, washdown)
- Visual: Red/brown staining, pitting with regular spacing matching rolling element pitch
- Vibration signature: Early kurtosis increase, irregular impacts
- Detection: High kurtosis (> 4.0) with erratic behavior

**Fretting Corrosion**
- Root cause: Micro-movement between bearing bore and shaft, insufficient interference fit
- Visual: Red-brown oxide powder at mating surfaces
- Vibration signature: 1x shaft frequency vibration, generally low kurtosis

---

### Category 4: Electrical Erosion (Fluting)
- Root cause: Stray electrical current through bearing (from VFD/inverter drives)
- Visual: Regular circumferential grooves (fluting) on raceway, washboard pattern
- Vibration signature: Regular high-frequency noise, harmonic series
- Prevention: Insulated bearing or shaft grounding brush

---

## 3. Severity Progression Model

For outer race spalling (dominant IMS failure mode), four stages are recognized:

| Stage | RMS (typical) | Kurtosis | Time to Failure | Action |
|---|---|---|---|---|
| **Stage 1** (Normal) | < 0.05 g | 2.5 – 3.5 | Months | Normal operation |
| **Stage 2** (Early fault) | 0.05–0.15 g | 3.5 – 5.0 | Weeks | Increase monitoring to daily |
| **Stage 3** (Developing) | 0.15–0.50 g | 5.0 – 8.0 | Days | Schedule maintenance ASAP |
| **Stage 4** (Advanced) | > 0.50 g | > 8.0 | Hours | IMMEDIATE replacement |

Note: IMS dataset uses acceleration (g) directly from sensor. The ISO 10816
velocity thresholds (mm/s) require integration of the acceleration signal.

---

## 4. Root Cause Analysis Checklist

When a bearing shows elevated vibration, check in order:

1. **Lubrication condition**: Is grease correct grade, quantity, and freshness?
2. **Contamination**: Any signs of water, dust, or process fluid ingress past seals?
3. **Alignment**: Is shaft alignment within 0.05 mm/m tolerance?
4. **Load**: Is radial load within bearing design limit (C/P ratio > 3)?
5. **Speed**: Is operating speed within bearing design speed limit?
6. **Temperature**: Is bearing running hot (> 80°C surface temperature)?
7. **Electrical**: Is drive system generating common-mode currents?

---

## 5. Corrective Action Decision Tree

```
High vibration detected (Zone C or D)
          |
    Is it Zone D? (RMS > 7.1 mm/s)
          |
    YES: STOP MACHINE IMMEDIATELY
    NO:  Continue to investigation
          |
    Kurtosis > 5.0?
          |
    YES: Bearing fault (impulsive events) -> Inspect bearing
    NO:  Possible imbalance/misalignment -> Check alignment, balance
          |
    BPFO peak in FFT?
          |
    YES: Outer race spalling confirmed -> Replace bearing
    NO:  Check BPFI (inner race), BSF (ball), FTF (cage)
```

---

## 6. IMS Bearing Replacement Recommendations

Based on dataset analysis:

| Test | Bearing | Recommended Replacement Hour |
|---|---|---|
| Test 1 | Bearing 3 | ~750 hrs (kurtosis > 4.5 in x-channel) |
| Test 1 | Bearing 4 | ~780 hrs (RMS crossing 0.2 g threshold) |
| Test 2 | Bearing 1 | ~152 hrs (final 12 hrs — rapid degradation) |
| Test 3 | Bearing 3 | ~900 hrs (RMS begins sustained increase) |
