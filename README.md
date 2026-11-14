# Burning Activity Calendar (MODIS + VIIRS)

Calendario de actividad de quema (meses × años) para cualquier área, a partir de hotspots NASA FIRMS (MODIS 2000+ y VIIRS 2012+) armonizados en una sola serie. Piloto: Colombia.

Fuente de datos: NASA FIRMS. Los años 2000–2011 son estimaciones (MODIS convertido a escala VIIRS); la serie termina el 2026-11-01 porque VIIRS S-NPP dejó de entregar datos.

Ejecutar localmente: `pip install -r requirements.txt` y `BURNCAL_DATA=data/processed python -m uvicorn app.main:app --port 8000`.
