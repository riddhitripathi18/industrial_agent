# TEMA Standards for Shell-and-Tube Heat Exchangers

## Source
TEMA (Tubular Exchanger Manufacturers Association) Standards, 9th Edition.
Applicable to shell-and-tube heat exchangers in refinery and petrochemical service.

---

## 1. Fouling Resistance (Rf) Design Values

Fouling resistance Rf is the key maintenance trigger for heat exchangers.
Units: m²·K/W (also expressed as hr·ft²·°F/BTU in imperial).

### Crude Oil Service (Tube Side)
| Crude Type | Rf (m²·K/W) | Cleaning Trigger |
|---|---|---|
| Light crude (API > 35) | 0.00035 | Rf > 0.00035 |
| Medium crude (API 25–35) | 0.00044 | Rf > 0.00044 |
| Heavy crude (API < 25) | 0.00053 | Rf > 0.00053 |
| **Standard design value (all crude)** | **0.00050** | **Rf > 0.00050** |

### Shell-Side Process Fluids
| Fluid | Rf (m²·K/W) |
|---|---|
| Heavy Naphtha | 0.00020 |
| Kerosene | 0.00020 |
| Light Diesel | 0.00035 |
| Heavy Diesel | 0.00035 |
| LVGO (Light Vacuum Gas Oil) | 0.00044 |
| HVGO (Heavy Vacuum Gas Oil) | 0.00053 |
| Steam (clean) | 0.00009 |
| Cooling Water (treated) | 0.00017 |

---

## 2. TEMA Exchanger Classifications

| Class | Application | Design Standard |
|---|---|---|
| **R** | Severe petroleum refinery service | Most stringent — highest wall thickness, corrosion allowance |
| **C** | General commercial & process | Moderate requirements |
| **B** | Chemical process service | Intermediate requirements |

Crude preheat train exchangers fall under **TEMA Class R**.

---

## 3. Cleaning Trigger Decision Logic

An exchanger requires cleaning when ANY of the following occur:
1. **Rf exceeds TEMA limit**: Measured Rf > 0.0005 m²·K/W for crude service
2. **U degradation > 25%**: U_dirty < 0.75 × U_clean (baseline)
3. **Outlet temperature drop > 8°C**: Sustained over 72+ hours
4. **Pressure drop increase > 50%**: Indicates heavy deposit or partial blockage
5. **Energy penalty > $500/hr**: Unrecovered heat exceeds economic threshold

---

## 4. Heat Transfer Area and U-Values

### Typical Overall U Ranges (Shell-and-Tube, Crude Service)
| Fluid Combination | U (W/m²·K) |
|---|---|
| Crude oil vs. Light hydrocarbon | 200–400 |
| Crude oil vs. Heavy hydrocarbon | 150–300 |
| Crude oil vs. Steam | 500–800 |
| Crude oil vs. Cooling water | 300–600 |

### LMTD Correction Factor (F)
For multi-pass configurations, U_eff = U × F where:
- 1-2 shell-and-tube (1 shell pass, 2 tube passes): F ≈ 0.85–0.97
- Counter-current pure: F = 1.0 (used as approximation in this system)

---

## 5. Recommended Cleaning Methods

| Fouling Type | Recommended Method |
|---|---|
| Asphaltene / heavy organic deposits | Chemical solvent cleaning (aromatic solvent wash) |
| Inorganic scale (calcium, iron sulfide) | Chemical acid cleaning (HCl 5–10%) |
| Light hydrocarbon residue | High-pressure hot water jetting (700–1000 bar) |
| Mixed fouling (organic + scale) | Sequential: chemical soak then hydroblast |
| Biological fouling (cooling water) | Chlorination or biocide dosing |

**Warning**: Never use mechanical rodding on U-tube bundles — risk of tube damage.

---

## 6. Inspection and Maintenance Intervals

| Fouling Severity | Inspection Interval | Cleaning Interval |
|---|---|---|
| Low (Rf < 0.0003) | 12 months | 24–36 months |
| Moderate (Rf 0.0003–0.0005) | 6 months | 12–18 months |
| High (Rf > 0.0005) | **Immediate** | **Within 4 weeks** |

---

## 7. Economic Justification for Cleaning

Cleaning decision economics:
- **Cleaning cost** (typical): $15,000–$80,000 per exchanger (chemical + labour + downtime)
- **Heat loss cost**: $200–$800/hr depending on fuel price and duty loss
- **Break-even**: Cleaning justified when cumulative energy loss > cleaning cost
- **Formula**: Hours_to_breakeven = Cleaning_cost_USD / Cost_per_hr_USD
