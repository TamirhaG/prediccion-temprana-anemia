import json
from anemia_model.predict import predict_one

def test_predict_smoke():
    payload = {
        "Departamento": "Lima",
        "Provincia": "Lima",
        "Distrito": "San Juan de Lurigancho",
        "Altitud_m": 250,
        "Edad_meses": 24,
        "Sexo": "M",
        "Area": "Urbana",
        "Ingreso_Familiar_Soles": 900,
        "Nro_Hijos": 2,
        "Nivel_Educacion_Madre": "Secundaria",
        "Actividad_Madre": "Comerciante",
        "Condicion_Vivienda": "Alquilada",
        "Acceso_Informacion": "Televisión",
        "Programa_QaliWarma": "No",
        "Programa_Juntos": "Sí",
        "Programa_VasoLeche": "Sí",
        "Suplemento_Hierro": "Sí",
        "Lugar_Atencion": "Centro de salud",
        "Peso_kg": 11.2,
        "Talla_cm": 84.0,
        "IMC_Infantil": 15.8,
        "Estado_Nutricional": "Normal",
        "Bajo_Peso": "No",
        "Talla_Baja": "No",
        "Hemoglobina_g_dL": 11.1,
        "Hemoglobina_Ajustada": 10.9
    }
    label, proba = predict_one(payload)
    assert label in {"No","Leve","Moderada","Severa"}
    assert isinstance(proba, dict)
