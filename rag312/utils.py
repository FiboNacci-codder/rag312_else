"""Utilidades genéricas compartidas por ingesta/ y consulta/."""


def formatear_tiempo(segundos: float) -> str:
    minutos = int(segundos // 60)
    segs = segundos % 60
    return f"{minutos}m {segs:.1f}s"
