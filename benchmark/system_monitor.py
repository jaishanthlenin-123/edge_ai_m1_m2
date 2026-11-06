"""
System monitoring utility for Raspberry Pi 5.
Reads real-time hardware telemetry: CPU utilization, per-core loads,
process RAM, system memory, SoC thermal status, and throttling flags.
"""

import os
import subprocess
import psutil

def get_cpu_temperature():
    """Reads Raspberry Pi SoC temperature in Celsius."""
    # First try Linux sysfs thermal zone
    thermal_path = "/sys/class/thermal/thermal_zone0/temp"
    if os.path.exists(thermal_path):
        try:
            with open(thermal_path, "r") as f:
                return float(f.read().strip()) / 1000.0
        except Exception:
            pass

    # Fallback to vcgencmd
    try:
        res = subprocess.check_output(["vcgencmd", "measure_temp"], text=True)
        # Output format: temp=52.0'C
        val = res.strip().replace("temp=", "").replace("'C", "")
        return float(val)
    except Exception:
        return None


def get_throttled_state():
    """
    Decodes Raspberry Pi vcgencmd get_throttled output.
    Returns:
        dict with raw_hex, is_throttled (bool), and list of active condition strings.
    """
    try:
        res = subprocess.check_output(["vcgencmd", "get_throttled"], text=True)
        # Format: throttled=0x0
        raw_str = res.strip().split("=")[-1]
        val = int(raw_str, 16)
    except Exception:
        return {"raw_hex": "N/A", "is_throttled": False, "flags": ["Unavailable"]}

    flags = []
    if val & 0x1:
        flags.append("Under-voltage detected")
    if val & 0x2:
        flags.append("ARM frequency capped")
    if val & 0x4:
        flags.append("Currently throttled")
    if val & 0x8:
        flags.append("Soft temperature limit active")
    if val & 0x10000:
        flags.append("Under-voltage has occurred")
    if val & 0x20000:
        flags.append("ARM frequency capping has occurred")
    if val & 0x40000:
        flags.append("Throttling has occurred")
    if val & 0x80000:
        flags.append("Soft temperature limit has occurred")

    if not flags:
        flags.append("None (Optimal Operating Condition)")

    return {
        "raw_hex": raw_str,
        "is_throttled": bool(val & 0x4) or bool(val & 0x2) or bool(val & 0x1),
        "flags": flags
    }


def get_system_telemetry(process=None):
    """
    Collects a comprehensive telemetry snapshot:
    CPU %, Per-Core CPU %, RAM (process and system), Temperature, Throttled state.
    """
    if process is None:
        process = psutil.Process()

    # Per-core utilization
    per_core = psutil.cpu_percent(interval=None, percpu=True)
    total_cpu = psutil.cpu_percent(interval=None)

    # Process RAM
    proc_mem = process.memory_info()
    proc_rss_mb = proc_mem.rss / (1024 * 1024)

    # System RAM
    sys_mem = psutil.virtual_memory()

    # Temperature and throttling
    temp_c = get_cpu_temperature()
    throttled = get_throttled_state()

    return {
        "cpu_total_percent": total_cpu,
        "cpu_per_core_percent": per_core,
        "process_rss_mb": proc_rss_mb,
        "system_ram_available_mb": sys_mem.available / (1024 * 1024),
        "system_ram_total_mb": sys_mem.total / (1024 * 1024),
        "system_ram_percent": sys_mem.percent,
        "temperature_c": temp_c,
        "throttled_hex": throttled["raw_hex"],
        "is_throttled": throttled["is_throttled"],
        "throttled_flags": "; ".join(throttled["flags"])
    }
