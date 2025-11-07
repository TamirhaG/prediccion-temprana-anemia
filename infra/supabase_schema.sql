CREATE TABLE IF NOT EXISTS public.pacientes_anemia (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  fecha_registro timestamptz DEFAULT now(),
  edad_meses int4,
  sexo varchar(1),
  peso_kg float4,
  talla_cm float4,
  altitud_m float4,
  area_rural bool,
  ingreso_familiar_soles float4,
  nro_hijos int2,
  suplementacion_hierro bool,
  anemia varchar(20),
  prob_riesgo_modelo float4,
  riesgo_predicho varchar(20)
);
