# cargas_manuales

Documentos subidos manualmente vía la interfaz web (`POST /api/upload` en `interfaz/app_web.py`), procesados incrementalmente por `ingesta/ingestar_archivo.py`.

No mezclar archivos acá con la taxonomía curada de `procedimientos/` y `Registros de Gestión Comercial/` — esta carpeta existe justamente para mantenerlos separados. `categoria` para estos chunks queda como `"cargas_manuales"`.
