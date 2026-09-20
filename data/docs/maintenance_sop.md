# Heat Exchanger and Bearing Maintenance SOPs

## Scope
Standard Operating Procedures for maintenance activities on:
1. Crude oil preheat train heat exchangers (E01–E05)
2. Rolling element bearings on rotating machinery

---

# PART A: HEAT EXCHANGER CLEANING SOPs

## SOP-HX-001: Chemical Cleaning (Organic Fouling — Asphaltene / Naphthenic)

### Trigger Conditions
- Measured Rf > 0.0005 m²·K/W (TEMA limit breached)
- Crude outlet temperature drop > 8°C sustained over 72 hours
- Energy loss > $500/hr confirmed by calculations

### Safety Prerequisites
- Exchanger isolated from process: all block valves CLOSED
- Depressured to atmospheric pressure — verify with pressure gauge
- Drained and purged with nitrogen (LEL < 1% confirmed)
- Hot work permit obtained if solvent flash point < 60°C
- Spill containment bunds in place

### Procedure
1. **Pre-cleaning flush**: Circulate hot water (80°C) for 2 hours to soften deposits
2. **Solvent selection**: Aromatic solvent (toluene-based) for asphaltene; HVGO wash for heavy organic
3. **Circulation**: Pump solvent at 0.5–1.0 m/s through tube side for 6–12 hours
4. **Temperature**: Maintain 60–80°C solvent temperature for best asphaltene dissolution
5. **Flush**: Drain solvent, flush 3× with diesel or light crude to remove residue
6. **Final water flush**: Hot water flush until effluent is clear
7. **Inspection**: Borescope tube inspection to confirm cleanliness
8. **Leak test**: Hydrostatic test at 1.5× design pressure before returning to service

### Acceptance Criteria
- Post-cleaning U within 10% of U_clean baseline
- Rf < 0.0001 m²·K/W measured on first day after startup

---

## SOP-HX-002: Hydroblasting (Mixed / Inorganic Fouling)

### Trigger Conditions
- Scale deposits confirmed (iron sulfide, calcium carbonate, wax)
- Chemical cleaning ineffective (U not restored after chemical treatment)

### Safety Prerequisites
- All energy isolation completed (same as SOP-HX-001)
- Bundle removed from shell (requires crane lift)
- High-pressure water exclusion zone: minimum 3 meters radius
- Hydroblast operator: full face shield, cut-resistant gloves, waterproof suit

### Procedure
1. **Remove bundle**: Extract tube bundle per mechanical procedure MP-HX-01
2. **Initial rinse**: Low-pressure wash (50 bar) to remove loose deposits
3. **Hydroblast**: 700–1000 bar water jet, clean each tube individually
   - Nozzle speed: 0.3–0.5 m/s through tube length
   - Cover full tube cross-section with 3 overlapping passes
4. **Inspection**: UT (ultrasonic thickness) measurement on minimum 10% of tubes
5. **Plugging**: Plug any tubes with wall loss > 30% of original thickness
6. **Bundle reassembly**: Reinstall per MP-HX-02 torque specs

### Acceptance Criteria
- All tubes visually clear (borescope 100% on first 2 tube rows, 20% random on remainder)
- No tube wall loss > 30% of nominal wall thickness

---

## SOP-HX-003: Post-Cleaning Monitoring Protocol

After any cleaning intervention, perform enhanced monitoring:

| Day | Action |
|---|---|
| Day 1 | Record U, delta-T tube/shell for all 5 exchangers at 4-hour intervals |
| Day 2–7 | Daily U calculation and comparison vs U_clean baseline |
| Day 8–30 | Weekly Rf calculation |
| Day 31+ | Resume normal monitoring interval |

**Alert threshold post-cleaning**: Rf > 0.0002 m²·K/W within 30 days indicates
re-fouling at abnormal rate — investigate crude quality (TAN, chlorides, API gravity).

---

# PART B: BEARING MAINTENANCE SOPs

## SOP-BRG-001: Planned Bearing Replacement

### Trigger Conditions (any one is sufficient)
- RMS velocity reaches ISO Zone C (> 4.5 mm/s) with confirmed upward trend
- Kurtosis sustained above 5.0 for > 24 hours
- RUL estimate < 100 hours based on trend extrapolation
- Scheduled replacement interval reached (typically 12,000–20,000 operating hours)

### Tools Required
- Bearing puller (hydraulic, rated for shaft diameter)
- Induction bearing heater
- Dial indicator and magnetic base
- Torque wrench (calibrated)
- Infrared thermometer
- Clean rags and solvent (IPA or acetone)
- Correct replacement bearing (verify SKF/FAG part number)
- Fresh grease (compatible grade — check manufacturer specs)

### Procedure
1. **Isolation**: De-energise motor, LOTO (Lockout-Tagout) applied and verified
2. **Disassembly**: Remove coupling, end covers, seals in reverse assembly order
3. **Bearing removal**: Use hydraulic puller — NEVER hammer directly on bearing
4. **Shaft inspection**: Check for fretting, scoring, ovality (max 0.02 mm runout)
5. **Housing inspection**: Check bore for fretting, corrosion, correct size (mic measurement)
6. **New bearing installation**:
   - Heat bearing to 80–100°C using induction heater (NEVER open flame)
   - Slide onto shaft immediately while hot
   - Push to correct position against shoulder (confirm with feeler gauge)
7. **Greasing**: Fill 30–50% of free space with correct grade grease (do NOT overfill)
8. **Reassembly**: Rebuild in reverse order, torque all fasteners per specification
9. **Alignment**: Check and correct coupling alignment (< 0.05 mm/m angular, < 0.1 mm parallel)
10. **Commissioning**: Start at low load, monitor vibration for 1 hour before full load

### Acceptance Criteria
- Vibration < 2.3 mm/s (ISO Zone A) within 30 minutes of startup
- Bearing temperature < 70°C during first 4 hours of operation

---

## SOP-BRG-002: Emergency Bearing Change (Zone D — Machine Tripped)

### When to Use
Machine has reached Zone D (RMS > 7.1 mm/s) or auto-tripped on high vibration.

### Additional Precautions vs SOP-BRG-001
- Inspect for secondary damage (shaft bending, housing cracking, seal damage)
- Collect failed bearing for failure analysis — do not clean before analysis
- Photograph damage before disassembly
- Notify maintenance engineer and reliability engineer before restart
- If shaft deflection > 0.1 mm, do NOT restart — replace shaft

### Failure Analysis Requirements
- Complete ISO 15243 failure classification form
- Photograph raceway, cage, rolling elements at 5× minimum magnification
- Record operating history: last greasing date, vibration history, process conditions
- Submit failed bearing to bearing OEM for detailed metallurgical analysis if failure cause unknown

---

## SOP-BRG-003: Lubrication Interval and Grease Specification

### Recommended Greasing Interval
| Operating Speed | Temperature | Interval |
|---|---|---|
| < 1500 RPM | < 60°C | 6 months |
| 1500–3000 RPM | < 70°C | 3 months |
| > 3000 RPM | < 80°C | 1 month |
| Any speed | > 80°C bearing temperature | Inspect immediately |

### Grease Selection (IMS Rig Equivalent)
- Grade: NLGI 2 lithium complex or polyurea grease
- Base oil viscosity: ISO VG 100–150 at 40°C
- Operating temperature range: -20°C to +150°C
- Incompatible with: calcium soap greases (DO NOT MIX)

**Critical**: When changing grease type, flush old grease completely with compatible
base oil before applying new grease to prevent incompatibility reactions.
