# ISO 10816 Vibration Severity Standards for Rotating Machinery

## Source
ISO 10816-3:2009 — Mechanical vibration: Evaluation of machine vibration by measurements
on non-rotating parts. Part 3: Industrial machines with nominal power above 15 kW and
nominal speeds between 120–15,000 RPM.

---

## 1. Vibration Severity Zones

ISO 10816 classifies vibration severity into four zones based on broadband RMS
velocity (mm/s, measured at bearing housings):

| Zone | RMS Range (mm/s) | Description | Recommended Action |
|---|---|---|---|
| **A** | 0.0 – 2.3 | **GOOD** — Newly commissioned or recently overhauled machine | No action. Normal operation. |
| **B** | 2.3 – 4.5 | **ACCEPTABLE** — Suitable for unrestricted long-term operation | No action. Schedule routine inspection. |
| **C** | 4.5 – 7.1 | **RESTRICTED** — Marginal for long-term operation | Plan maintenance within 2–4 weeks. Monitor daily. |
| **D** | > 7.1 | **DANGER** — Risk of immediate machine damage | **STOP MACHINE IMMEDIATELY**. Do not restart until cause is identified and corrected. |

> **Critical Note**: Zone D machines pose risk of catastrophic failure, safety hazards,
> secondary equipment damage, and personnel injury. Never defer Zone D corrective action.

---

## 2. Machine Class Definitions (ISO 10816-3)

| Class | Machine Type | Examples |
|---|---|---|
| **Group 1** | Large machines (> 300 kW) on rigid foundations | Large pumps, compressors, turbines |
| **Group 2** | Medium machines (15–300 kW) | Process pumps, fans, centrifuges |
| **Group 3** | Pumps with separate driver | Centrifugal pumps in petroleum service |
| **Group 4** | Pumps with integrated driver | Submersible, inline pumps |

**IMS Bearing Test Rig**: Classified as Group 2 (15 kW AC motor, 2000 RPM).

---

## 3. Measurement Locations and Axes

Per ISO 10816, measurements must be taken at:
- **Bearing housings** (not casing, not base plate)
- **Three orthogonal directions**: horizontal (H), vertical (V), axial (A)
- **Both inboard and outboard bearings** for each shaft

For the IMS dataset:
- Channels labelled `_x` = horizontal (radial) axis
- Channels labelled `_y` = vertical (radial) axis
- Most sensitive axis for outer/inner race defects: **radial (x or y)**
- Most sensitive axis for shaft misalignment: **axial**

---

## 4. Alert and Trip Setpoints (Common Industry Practice)

Beyond ISO zone boundaries, plants set machine-specific alert/trip levels:

| Setpoint | Typical Value | Consequence |
|---|---|---|
| **Alert (early warning)** | Zone B/C boundary: 4.5 mm/s | Operator notification, increase monitoring frequency |
| **Danger (auto-trip)** | Zone C/D boundary: 7.1 mm/s | Automatic machine shutdown initiated |
| **Custom high-vibration trip** | 10–15 mm/s | Immediate emergency stop |

---

## 5. Vibration Trend Interpretation

### Gradual Degradation (fatigue wear)
- RMS increases slowly over weeks/months
- Kurtosis remains near baseline (2.5–3.5) until late stage
- Indicates: normal bearing wear, lubrication degradation
- Action: plan maintenance at next turnaround

### Sudden Onset (impact event)
- RMS spikes rapidly within hours or days
- Kurtosis jumps > 4.0 (impulsive events from micro-cracks)
- Indicates: foreign material ingress, overload, installation damage
- Action: immediate inspection

### Periodic Spikes (resonance / imbalance)
- Vibration fluctuates with process load
- FFT shows dominant component at shaft frequency (1x) or harmonics
- Indicates: imbalance, misalignment, looseness
- Action: dynamic balancing or alignment check at next shutdown

---

## 6. Bearing Fault Frequencies

For IMS test rig (shaft speed ~2000 RPM = 33.33 Hz, 6205-2RS SKF bearing):

| Fault Type | Frequency | Formula |
|---|---|---|
| **BPFO** — Ball Pass Frequency, Outer Race | ~236 Hz | (Nb/2) x RPM/60 x (1 - Bd/Pd x cos(a)) |
| **BPFI** — Ball Pass Frequency, Inner Race | ~297 Hz | (Nb/2) x RPM/60 x (1 + Bd/Pd x cos(a)) |
| **BSF** — Ball Spin Frequency | ~139 Hz | (Pd/(2xBd)) x RPM/60 x (1 - (Bd/Pd x cos(a))^2) |
| **FTF** — Fundamental Train Frequency | ~13 Hz | (RPM/60)/2 x (1 - Bd/Pd x cos(a)) |

Parameters: Nb=8 balls, Bd=0.331" ball diameter, Pd=1.318" pitch diameter, a=0 deg contact angle.

A spike at BPFO in the FFT is the primary indicator of outer race spalling,
which is the most common failure mode in the IMS Test 2 and Test 3 datasets.

---

## 7. Lubrication and Its Effect on Vibration

| Lubrication Condition | Vibration Effect | Kurtosis |
|---|---|---|
| Correct (grease quantity and grade) | Low baseline RMS, stable trend | 2.5 to 3.5 |
| Under-lubricated | Progressive RMS increase, metal-to-metal contact | Increases > 4.0 |
| Over-lubricated | Initial spike then settling, heat generation | Temporary spike |
| Contaminated (water or debris ingress) | Erratic RMS, chemical corrosion of raceway | Highly variable |

IMS Test 1 lubricant failure: Both channels of Bearing 3 and Bearing 4 show
progressive RMS increase consistent with lubricant starvation and outer race spalling.
