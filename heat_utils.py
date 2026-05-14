import numpy as np


MILD_STEEL_PROPERTIES = {
    "density_kg_per_m3": 7850.0,
    "specific_heat_j_per_kgk": 500.0,
    "melting_temperature_c": 1500.0,
    "initial_temperature_c": 25.0,
    "latent_heat_fusion_j_per_kg": 272000.0,
}

DEFAULT_HEAT_LOSS_FRACTIONS = {
    "Conduction to surrounding plate": 0.35,
    "Convection to surroundings": 0.20,
    "Radiation loss": 0.15,
    "Spatter and fume loss": 0.15,
    "Torch, electrode, and fixture loss": 0.15,
}

PULSE_CATEGORY_ORDER = ["Normal", "Effective", "Arcing", "Shorts"]

PULSE_CATEGORY_REFERENCE = {
    "Normal": {
        "band": "peak pulse current / I0 >= 0.72",
        "description": "strong current pulse above the normal threshold",
    },
    "Effective": {
        "band": "0.65 <= peak pulse current / I0 < 0.72",
        "description": "effective current pulse, but weaker than the normal pulse",
    },
    "Arcing": {
        "band": "0.55 <= peak pulse current / I0 < 0.65",
        "description": "weak current pulse in the arcing range",
    },
    "Shorts": {
        "band": "peak pulse current / I0 < 0.55",
        "description": "very weak or short current pulse below the short threshold",
    },
}

HEAT_LOSS_MECHANISMS = {
    "Useful heat to weld pool": {
        "share_label": "Remaining useful heat",
        "mechanism": (
            "This is the portion of electrical energy that actually enters the weld pool and base material."
        ),
        "effect": (
            "It raises temperature, melts metal, and contributes directly to fusion and bead formation."
        ),
    },
    "Conduction to surrounding plate": {
        "share_label": "35% of total losses",
        "mechanism": (
            "Heat flows away from the hot weld zone into the colder parent metal because of the temperature gradient."
        ),
        "effect": (
            "This lowers the weld pool temperature and is usually the largest loss path in steel plates."
        ),
    },
    "Convection to surroundings": {
        "share_label": "20% of total losses",
        "mechanism": (
            "Hot gases, shielding gas, and surrounding air carry heat away from the arc and exposed weld surface."
        ),
        "effect": (
            "This removes surface heat continuously, especially from the arc plume and hot bead."
        ),
    },
    "Radiation loss": {
        "share_label": "15% of total losses",
        "mechanism": (
            "The arc column and incandescent metal radiate thermal energy as electromagnetic waves."
        ),
        "effect": (
            "Radiative loss becomes important at very high arc and pool temperatures."
        ),
    },
    "Spatter and fume loss": {
        "share_label": "15% of total losses",
        "mechanism": (
            "Molten droplets, expelled particles, and vaporized metal leave the weld zone carrying thermal energy with them."
        ),
        "effect": (
            "This reduces usable heat and also decreases material deposition efficiency."
        ),
    },
    "Torch, electrode, and fixture loss": {
        "share_label": "15% of total losses",
        "mechanism": (
            "Part of the input energy is absorbed by the electrode, contact tip, torch body, clamps, and fixtures."
        ),
        "effect": (
            "That energy heats the tooling system instead of the weld pool."
        ),
    },
}


def _coerce_signal(signal, sample_count: int, signal_name: str):
    signal = np.asarray(signal, dtype=float)
    if signal.ndim == 0:
        return np.full(sample_count, float(signal))

    signal = signal.reshape(-1)
    if signal.size != sample_count:
        raise ValueError(f"{signal_name} must be a scalar or have the same number of samples as time_ms.")
    return signal


def _extract_true_runs(mask):
    runs = []
    run_start = None

    for index, is_true in enumerate(mask):
        if is_true and run_start is None:
            run_start = index
        elif not is_true and run_start is not None:
            runs.append((run_start, index))
            run_start = None

    if run_start is not None:
        runs.append((run_start, mask.size))

    return runs


def _fill_short_false_gaps(mask, max_gap: int):
    if max_gap <= 0:
        return mask

    refined_mask = mask.astype(bool).copy()
    false_runs = _extract_true_runs(~refined_mask)

    for start, end in false_runs:
        if start == 0 or end == refined_mask.size:
            continue
        if (end - start) <= max_gap:
            refined_mask[start:end] = True

    return refined_mask


def _remove_short_true_runs(mask, min_run_length: int):
    if min_run_length <= 1:
        return mask

    refined_mask = mask.astype(bool).copy()
    true_runs = _extract_true_runs(refined_mask)

    for start, end in true_runs:
        if (end - start) < min_run_length:
            refined_mask[start:end] = False

    return refined_mask


def _finite_mean(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.nan
    return float(np.mean(finite_values))


def _safe_pulse_average(values):
    finite_values = [value for value in values if np.isfinite(value)]
    if not finite_values:
        return np.nan
    return float(np.mean(finite_values))


def _classify_pulse_voltage_ratio(voltage_ratio: float) -> str:
    if not np.isfinite(voltage_ratio):
        return "Shorts"
    if voltage_ratio >= 0.72:
        return "Normal"
    if voltage_ratio >= 0.65:
        return "Effective"
    if voltage_ratio >= 0.55:
        return "Arcing"
    return "Shorts"


def _build_empty_pulse_summary():
    return {
        category_name: {
            "pulse_count": 0,
            "avg_duration_ms": np.nan,
            "avg_electrical_energy_j": np.nan,
            "avg_useful_heat_j": np.nan,
            "avg_temperature_rise_c": np.nan,
            "band": PULSE_CATEGORY_REFERENCE[category_name]["band"],
            "description": PULSE_CATEGORY_REFERENCE[category_name]["description"],
        }
        for category_name in PULSE_CATEGORY_ORDER
    }


def _detect_current_pulse_runs(time_ms, current_a):
    finite_current = current_a[np.isfinite(current_a)]
    low_current_level_a = float(np.percentile(finite_current, 25)) if finite_current.size else np.nan
    high_current_level_a = float(np.percentile(finite_current, 80)) if finite_current.size else np.nan
    dynamic_span_a = high_current_level_a - low_current_level_a if np.isfinite(high_current_level_a) else np.nan
    pulse_threshold_a = (
        low_current_level_a + 0.45 * dynamic_span_a
        if np.isfinite(dynamic_span_a) and dynamic_span_a > 1e-9
        else np.nan
    )

    active_mask = np.zeros(time_ms.size, dtype=bool)
    if np.isfinite(pulse_threshold_a):
        active_mask = np.isfinite(current_a) & (current_a >= pulse_threshold_a)
        min_pulse_samples = max(2, int(np.ceil(0.002 * time_ms.size)))
        max_gap_samples = max(1, int(np.ceil(0.001 * time_ms.size)))
        active_mask = _fill_short_false_gaps(active_mask, max_gap_samples)
        active_mask = _remove_short_true_runs(active_mask, min_pulse_samples)

    return low_current_level_a, high_current_level_a, pulse_threshold_a, _extract_true_runs(active_mask)


def analyze_welding_pulses(
    time_ms,
    current_a,
    voltage_v,
    efficiency=0.8,
    welding_speed_mm_per_s=1.0,
    weld_area_mm2=1.0,
    density_kg_per_m3=MILD_STEEL_PROPERTIES["density_kg_per_m3"],
    specific_heat_j_per_kgk=MILD_STEEL_PROPERTIES["specific_heat_j_per_kgk"],
    initial_temperature_c=MILD_STEEL_PROPERTIES["initial_temperature_c"],
):
    """
    Detect current pulses and compute per-pulse heat and temperature-rise metrics.

    Pulse discrimination follows the uploaded threshold logic but is applied to current:
    charging points are detected from low-current flags below 0.55 I0, each pulse is the
    waveform between two consecutive charging points, and the pulse type is assigned from
    the peak current in that interval relative to the reference current I0.
    """
    time_ms = np.asarray(time_ms, dtype=float).reshape(-1)
    current_a = np.asarray(current_a, dtype=float).reshape(-1)

    if time_ms.size == 0:
        raise ValueError("time_ms must contain at least one sample.")
    if current_a.size != time_ms.size:
        raise ValueError("current_a must have the same number of samples as time_ms.")

    voltage_v = _coerce_signal(voltage_v, time_ms.size, "voltage_v")
    efficiency = float(np.clip(efficiency, 0.0, 1.0))
    welding_speed_mm_per_s = float(welding_speed_mm_per_s)
    weld_area_mm2 = float(weld_area_mm2)

    dt_seconds = np.diff(time_ms, prepend=time_ms[0]) / 1000.0
    dt_seconds = np.where(np.isfinite(dt_seconds), dt_seconds, 0.0)
    dt_seconds = np.clip(dt_seconds, 0.0, None)

    low_current_level_a, high_current_level_a, pulse_threshold_a, current_pulse_runs = _detect_current_pulse_runs(
        time_ms, current_a
    )
    power_w = current_a * voltage_v
    reference_current_a = float(np.nanmax(current_a)) if np.any(np.isfinite(current_a)) else np.nan
    current_variation_a = float(np.nanstd(current_a)) if np.any(np.isfinite(current_a)) else np.nan
    short_threshold_a = 0.55 * reference_current_a if np.isfinite(reference_current_a) and reference_current_a > 0 else np.nan
    effective_threshold_a = 0.65 * reference_current_a if np.isfinite(reference_current_a) and reference_current_a > 0 else np.nan
    normal_threshold_a = 0.72 * reference_current_a if np.isfinite(reference_current_a) and reference_current_a > 0 else np.nan

    classification_available = bool(np.isfinite(reference_current_a) and reference_current_a > 0)
    charging_point_indices = []
    pulse_runs = []
    detection_method = "Current-flag pulse discrimination from consecutive charging points"

    if classification_available:
        flag_mask = np.isfinite(current_a) & (current_a <= short_threshold_a)
        max_gap_samples = max(1, int(np.ceil(0.001 * time_ms.size)))
        flag_mask = _fill_short_false_gaps(flag_mask, max_gap_samples)
        flag_runs = _extract_true_runs(flag_mask)

        for start, end in flag_runs:
            current_segment = current_a[start:end]
            if current_segment.size == 0 or not np.any(np.isfinite(current_segment)):
                continue
            charging_point_indices.append(start + int(np.nanargmin(current_segment)))

        charging_point_indices = sorted(set(charging_point_indices))
        for left_index, right_index in zip(charging_point_indices[:-1], charging_point_indices[1:]):
            if right_index - left_index >= 2:
                pulse_runs.append((left_index, right_index + 1))

        if pulse_runs:
            detection_method = "Current-flag pulse discrimination from consecutive charging points"
        else:
            detection_method = "Current-threshold pulse detection fallback"

    if not pulse_runs:
        pulse_runs = current_pulse_runs

    pulses = []
    for pulse_id, (start_index, end_index) in enumerate(pulse_runs, start=1):
        pulse_slice = slice(start_index, end_index)
        pulse_duration_s = float(np.sum(dt_seconds[pulse_slice]))
        pulse_duration_ms = pulse_duration_s * 1000.0
        electrical_energy_j = float(np.nansum(power_w[pulse_slice] * dt_seconds[pulse_slice]))
        useful_heat_j = float(efficiency * electrical_energy_j)
        pulse_mean_current_a = _finite_mean(current_a[pulse_slice])
        pulse_peak_current_a = float(np.nanmax(current_a[pulse_slice])) if np.any(np.isfinite(current_a[pulse_slice])) else np.nan
        pulse_mean_voltage_v = _finite_mean(voltage_v[pulse_slice])

        current_segment = current_a[pulse_slice]
        if np.any(np.isfinite(current_segment)):
            discharge_offset = int(np.nanargmax(current_segment))
            discharging_point_index = start_index + discharge_offset
            pulse_peak_current_a = float(current_a[discharging_point_index])
        else:
            discharge_offset = 0
            discharging_point_index = start_index + discharge_offset
            pulse_peak_current_a = np.nan

        pulse_current_ratio = (
            pulse_peak_current_a / reference_current_a
            if classification_available and np.isfinite(pulse_peak_current_a) and np.isfinite(reference_current_a) and reference_current_a > 0
            else np.nan
        )
        pulse_category = _classify_pulse_voltage_ratio(pulse_current_ratio) if classification_available else "Unclassified"

        pulse_length_mm = (
            welding_speed_mm_per_s * pulse_duration_s
            if np.isfinite(welding_speed_mm_per_s) and welding_speed_mm_per_s > 0
            else np.nan
        )
        pulse_volume_mm3 = (
            weld_area_mm2 * pulse_length_mm
            if np.isfinite(weld_area_mm2) and weld_area_mm2 > 0 and np.isfinite(pulse_length_mm)
            else np.nan
        )
        pulse_volume_m3 = pulse_volume_mm3 * 1e-9 if np.isfinite(pulse_volume_mm3) and pulse_volume_mm3 > 0 else np.nan
        pulse_mass_kg = (
            density_kg_per_m3 * pulse_volume_m3
            if np.isfinite(pulse_volume_m3) and pulse_volume_m3 > 0
            else np.nan
        )

        if np.isfinite(pulse_mass_kg) and pulse_mass_kg > 0 and specific_heat_j_per_kgk > 0:
            pulse_temperature_rise_c = useful_heat_j / (pulse_mass_kg * specific_heat_j_per_kgk)
            pulse_final_temperature_c = initial_temperature_c + pulse_temperature_rise_c
        else:
            pulse_temperature_rise_c = np.nan
            pulse_final_temperature_c = np.nan

        pulses.append(
            {
                "pulse_id": pulse_id,
                "start_time_ms": float(time_ms[start_index]),
                "end_time_ms": float(time_ms[end_index - 1]),
                "duration_ms": pulse_duration_ms,
                "charging_point_time_ms": float(time_ms[start_index]),
                "discharging_point_time_ms": float(time_ms[discharging_point_index]),
                "mean_current_a": pulse_mean_current_a,
                "peak_current_a": pulse_peak_current_a,
                "mean_voltage_v": pulse_mean_voltage_v,
                "current_ratio_i0": pulse_current_ratio,
                "electrical_energy_j": electrical_energy_j,
                "useful_heat_j": useful_heat_j,
                "pulse_length_mm": pulse_length_mm,
                "pulse_volume_mm3": pulse_volume_mm3,
                "pulse_mass_kg": pulse_mass_kg,
                "temperature_rise_c": pulse_temperature_rise_c,
                "final_temperature_c": pulse_final_temperature_c,
                "pulse_category": pulse_category,
            }
        )

    summary = _build_empty_pulse_summary()
    for category_name in PULSE_CATEGORY_ORDER:
        category_pulses = [pulse for pulse in pulses if pulse["pulse_category"] == category_name]
        summary[category_name] = {
            "pulse_count": len(category_pulses),
            "avg_duration_ms": _safe_pulse_average([pulse["duration_ms"] for pulse in category_pulses]),
            "avg_electrical_energy_j": _safe_pulse_average([pulse["electrical_energy_j"] for pulse in category_pulses]),
            "avg_useful_heat_j": _safe_pulse_average([pulse["useful_heat_j"] for pulse in category_pulses]),
            "avg_temperature_rise_c": _safe_pulse_average([pulse["temperature_rise_c"] for pulse in category_pulses]),
            "band": PULSE_CATEGORY_REFERENCE[category_name]["band"],
            "description": PULSE_CATEGORY_REFERENCE[category_name]["description"],
        }

    return {
        "total_pulses": len(pulses),
        "pulse_threshold_a": pulse_threshold_a,
        "low_current_level_a": low_current_level_a,
        "high_current_level_a": high_current_level_a,
        "reference_current_a": reference_current_a,
        "current_variation_a": current_variation_a,
        "short_threshold_a": short_threshold_a,
        "effective_threshold_a": effective_threshold_a,
        "normal_threshold_a": normal_threshold_a,
        "classification_available": classification_available,
        "charging_point_count": len(charging_point_indices),
        "avg_electrical_energy_j": _safe_pulse_average([pulse["electrical_energy_j"] for pulse in pulses]),
        "avg_useful_heat_j": _safe_pulse_average([pulse["useful_heat_j"] for pulse in pulses]),
        "avg_temperature_rise_c": _safe_pulse_average([pulse["temperature_rise_c"] for pulse in pulses]),
        "pulses": pulses,
        "summary": summary,
        "category_reference": PULSE_CATEGORY_REFERENCE,
        "classification_basis": "pulse peak current at the discharging point divided by I0",
        "detection_method": detection_method,
    }


def calculate_welding_heat(
    time_ms,
    current_a=None,
    voltage_v=None,
    power_w=None,
    efficiency=0.8,
    welding_speed_mm_per_s=1.0,
    weld_area_mm2=1.0,
    weld_length_mm=1.0,
):
    """
    Compute welding heat metrics from sampled electrical signals.

    Parameters
    ----------
    time_ms : array-like
        Sample timestamps in milliseconds.
    current_a : array-like, optional
        Current signal in amperes. Required when ``power_w`` is not supplied.
    voltage_v : float or array-like, optional
        Voltage in volts. Can be a scalar or a time-series with the same length as ``time_ms``.
        Required when ``power_w`` is not supplied.
    power_w : array-like, optional
        Precomputed power signal in watts. If supplied, it is used directly.
    efficiency : float, default=0.8
        Arc/process efficiency factor. Values are clipped into the physical range [0, 1].
    welding_speed_mm_per_s : float, default=1.0
        Travel speed in millimetres per second.
    weld_area_mm2 : float, default=1.0
        Weld cross-sectional area in square millimetres.
    weld_length_mm : float, default=1.0
        Weld length in millimetres.

    Returns
    -------
    dict
        Dictionary containing power, cumulative energy/heat, and scalar heat metrics.
    """
    time_ms = np.asarray(time_ms, dtype=float).reshape(-1)
    if time_ms.size == 0:
        raise ValueError("time_ms must contain at least one sample.")

    if power_w is None:
        if current_a is None or voltage_v is None:
            raise ValueError("Either power_w or both current_a and voltage_v must be provided.")

        current_a = np.asarray(current_a, dtype=float).reshape(-1)
        if current_a.size != time_ms.size:
            raise ValueError("current_a must have the same number of samples as time_ms.")

        voltage_v = np.asarray(voltage_v, dtype=float)
        if voltage_v.ndim == 0:
            voltage_v = np.full(time_ms.shape, float(voltage_v))
        else:
            voltage_v = voltage_v.reshape(-1)

        if voltage_v.size != time_ms.size:
            raise ValueError("voltage_v must be a scalar or have the same number of samples as time_ms.")

        power_w = voltage_v * current_a
    else:
        power_w = np.asarray(power_w, dtype=float).reshape(-1)
        if power_w.size != time_ms.size:
            raise ValueError("power_w must have the same number of samples as time_ms.")

        if voltage_v is None:
            voltage_v = np.full(time_ms.shape, np.nan)
        else:
            voltage_v = np.asarray(voltage_v, dtype=float)
            if voltage_v.ndim == 0:
                voltage_v = np.full(time_ms.shape, float(voltage_v))
            else:
                voltage_v = voltage_v.reshape(-1)
            if voltage_v.size != time_ms.size:
                raise ValueError("voltage_v must be a scalar or have the same number of samples as time_ms.")

        if current_a is None:
            current_a = np.full(time_ms.shape, np.nan)
        else:
            current_a = np.asarray(current_a, dtype=float).reshape(-1)
            if current_a.size != time_ms.size:
                raise ValueError("current_a must have the same number of samples as time_ms.")

    efficiency = float(np.clip(efficiency, 0.0, 1.0))
    welding_speed_mm_per_s = float(welding_speed_mm_per_s)
    weld_area_mm2 = float(weld_area_mm2)
    weld_length_mm = float(weld_length_mm)

    dt_seconds = np.diff(time_ms, prepend=time_ms[0]) / 1000.0
    dt_seconds = np.where(np.isfinite(dt_seconds), dt_seconds, 0.0)
    dt_seconds = np.clip(dt_seconds, 0.0, None)

    cumulative_energy_j = np.cumsum(power_w * dt_seconds)
    total_energy_j = float(cumulative_energy_j[-1]) if cumulative_energy_j.size else 0.0

    effective_power_w = efficiency * power_w
    cumulative_heat_j = np.cumsum(effective_power_w * dt_seconds)
    effective_heat_j = float(cumulative_heat_j[-1]) if cumulative_heat_j.size else 0.0
    total_loss_j = max(total_energy_j - effective_heat_j, 0.0)
    loss_breakdown_j = {
        name: total_loss_j * fraction for name, fraction in DEFAULT_HEAT_LOSS_FRACTIONS.items()
    }

    total_duration_s = float(np.sum(dt_seconds))
    travel_length_mm = welding_speed_mm_per_s * total_duration_s if welding_speed_mm_per_s > 0 else np.nan

    if np.isfinite(travel_length_mm) and travel_length_mm > 0:
        heat_input_per_length_j_per_mm = effective_heat_j / travel_length_mm
        instantaneous_heat_input_j_per_mm = effective_power_w / welding_speed_mm_per_s
    else:
        heat_input_per_length_j_per_mm = np.nan
        instantaneous_heat_input_j_per_mm = np.full_like(power_w, np.nan, dtype=float)

    volume_mm3 = weld_area_mm2 * weld_length_mm
    if volume_mm3 > 0:
        heat_density_j_per_mm3 = effective_heat_j / volume_mm3
    else:
        heat_density_j_per_mm3 = np.nan

    return {
        "time_ms": time_ms,
        "dt_seconds": dt_seconds,
        "current_a": current_a,
        "voltage_v": voltage_v,
        "power_w": power_w,
        "effective_power_w": effective_power_w,
        "cumulative_energy_j": cumulative_energy_j,
        "cumulative_heat_j": cumulative_heat_j,
        "instantaneous_heat_input_j_per_mm": instantaneous_heat_input_j_per_mm,
        "total_duration_s": total_duration_s,
        "travel_length_mm": travel_length_mm,
        "total_energy_j": total_energy_j,
        "effective_heat_j": effective_heat_j,
        "total_loss_j": total_loss_j,
        "loss_breakdown_j": loss_breakdown_j,
        "heat_input_per_length_j_per_mm": heat_input_per_length_j_per_mm,
        "heat_density_j_per_mm3": heat_density_j_per_mm3,
        "efficiency": efficiency,
        "weld_area_mm2": weld_area_mm2,
        "weld_length_mm": weld_length_mm,
    }


def material_analysis(
    effective_heat_j,
    weld_area_mm2,
    weld_length_mm,
    density_kg_per_m3=MILD_STEEL_PROPERTIES["density_kg_per_m3"],
    specific_heat_j_per_kgk=MILD_STEEL_PROPERTIES["specific_heat_j_per_kgk"],
    melting_temperature_c=MILD_STEEL_PROPERTIES["melting_temperature_c"],
    initial_temperature_c=MILD_STEEL_PROPERTIES["initial_temperature_c"],
    latent_heat_fusion_j_per_kg=MILD_STEEL_PROPERTIES["latent_heat_fusion_j_per_kg"],
):
    """
    Perform a mild-steel thermal sufficiency analysis for the supplied heat.

    Parameters
    ----------
    effective_heat_j : float
        Effective heat delivered to the weld in joules.
    weld_area_mm2 : float
        Weld cross-sectional area in square millimetres.
    weld_length_mm : float
        Weld length in millimetres.

    Returns
    -------
    dict
        Material/thermal properties and sufficiency results.
    """
    effective_heat_j = float(effective_heat_j)
    weld_area_mm2 = float(weld_area_mm2)
    weld_length_mm = float(weld_length_mm)

    volume_mm3 = weld_area_mm2 * weld_length_mm
    volume_m3 = volume_mm3 * 1e-9 if volume_mm3 > 0 else np.nan

    if np.isfinite(volume_m3) and volume_m3 > 0:
        mass_kg = density_kg_per_m3 * volume_m3
    else:
        mass_kg = np.nan

    if np.isfinite(mass_kg):
        sensible_heat_required_j = mass_kg * specific_heat_j_per_kgk * (
            melting_temperature_c - initial_temperature_c
        )
        latent_heat_required_j = mass_kg * latent_heat_fusion_j_per_kg
        required_heat_j = sensible_heat_required_j + latent_heat_required_j
    else:
        sensible_heat_required_j = np.nan
        latent_heat_required_j = np.nan
        required_heat_j = np.nan

    if np.isfinite(mass_kg) and mass_kg > 0 and specific_heat_j_per_kgk > 0:
        temperature_rise_c = effective_heat_j / (mass_kg * specific_heat_j_per_kgk)
        final_temperature_c = initial_temperature_c + temperature_rise_c
    else:
        temperature_rise_c = np.nan
        final_temperature_c = np.nan

    if np.isfinite(required_heat_j) and required_heat_j > 0:
        heat_sufficiency_pct = (effective_heat_j / required_heat_j) * 100.0
    elif abs(effective_heat_j) < 1e-12:
        heat_sufficiency_pct = 0.0
    else:
        heat_sufficiency_pct = np.nan

    melting_achieved = bool(np.isfinite(final_temperature_c) and final_temperature_c >= melting_temperature_c)
    melting_status = "Melting Achieved" if melting_achieved else "Not Achieved"

    return {
        "material_name": "Mild Steel",
        "density_kg_per_m3": density_kg_per_m3,
        "specific_heat_j_per_kgk": specific_heat_j_per_kgk,
        "melting_temperature_c": melting_temperature_c,
        "initial_temperature_c": initial_temperature_c,
        "latent_heat_fusion_j_per_kg": latent_heat_fusion_j_per_kg,
        "volume_mm3": volume_mm3,
        "volume_m3": volume_m3,
        "mass_kg": mass_kg,
        "sensible_heat_required_j": sensible_heat_required_j,
        "latent_heat_required_j": latent_heat_required_j,
        "required_heat_j": required_heat_j,
        "supplied_heat_j": effective_heat_j,
        "temperature_rise_c": temperature_rise_c,
        "final_temperature_c": final_temperature_c,
        "heat_sufficiency_pct": heat_sufficiency_pct,
        "melting_achieved": melting_achieved,
        "melting_status": melting_status,
    }
