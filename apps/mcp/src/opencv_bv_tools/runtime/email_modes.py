"""Existing email inference settings shared by the adapter and OCR worker."""
MODES = {
    "accurate": {"faint_pass": True, "gate_second_read": True},
    "fast": {"faint_pass": False, "gate_second_read": True},
}
