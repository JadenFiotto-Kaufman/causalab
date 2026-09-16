"""Triton kernels for the reference engine's grouped-experts path, each with
an order-explicit pure-torch reference of the ATen semantics it reproduces
(``moe_glue_reference.py``) and a device/shape plan that decides, per call,
which of them run (``moe_glue.py``)."""
