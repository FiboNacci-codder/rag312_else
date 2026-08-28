<!-- image -->

| INSTRUCTIVO ANÁLISIS DE INFORMACIÓN DE CLIENTES   | Código Versión Aprobado Fecha Página   | : CVPC-IN-002 : 00 : CSIG : 28-06-2018 : 1 de 8   |
|---------------------------------------------------|----------------------------------------|---------------------------------------------------|

| Nombre del procedimiento:   | Verificación de Suministros y Detección de Vulneración y/o Consumo sin Autorización   |
|-----------------------------|---------------------------------------------------------------------------------------|
| Código del procedimiento:   | CVPC-PR-001                                                                           |

8

## INSTRUCTIVO DE ANÁLISIS DE INFORMACIÓN DE CLIENTES

| Elaborado por:              | Revisado por:               | Homologado por:                                                       | Aprobado por:                           |
|-----------------------------|-----------------------------|-----------------------------------------------------------------------|-----------------------------------------|
| Jefe de Control de Pérdidas | Jefe de Control de Pérdidas | Especialista en Sistemas de Gestión Área de Planeamiento y Desarrollo | Comité del Sistema Integrado de Gestión |
| 17/05/2018                  | 06/06/2018                  | 18/06/2018                                                            | 28/06/2018                              |

<!-- image -->

## INSTRUCTIVO

## ANÁLISIS DE INFORMACIÓN DE CLIENTES

## 1. OBJETIVO Y ALCANCE

Establecer el instructivo para el análisis de la información de los clientes para la emisión del listado de clientes observados para las vistas de campo.

## 2. DEFINICIONES

- 2.1. Factor de transformación: Viene a ser el producto entre el factor de tensión y el factor de corriente, obtenido automáticamente del SIELSE.
- 2.2. Histórico de factor de transformación de facturación: son los factores de transformación que se aplican para cada ciclo de facturación.
- 2.3. Perfiles de carga: Comportamiento de la energía eléctrica en el tiempo (cada 15 minutos), para analizar detalladamente los consumos mensuales en cada ciclo de facturación.
- 2.4. POE: Plan Operativo Empresarial .

## 3. MARCO NORMATIVO Y/O DOCUMENTOS RELACIONADOS

- 3.1. CVPC-PR-001 Verificación de suministro y detección de vulneración y/o consumo sin autorización
- 3.2. Artículo 90º y 91º del Decreto Ley Nº 25844, Ley de Concesiones Eléctricas.
- 3.3. Artículos 177º, 202º, 204° y 205° del Reglamento de la Ley de Concesiones Eléctricas.
- 3.4. Normal DGE 'Reintegros y Recuperos de energía eléctrica' Nº 571-2006-MEM/DM.
- 3.5. Procedimiento para la supervisión de los reintegros y recuperos de energía eléctrica en el servicio público de electricidad de la RCD Nº 722-2007-OS/CD.
- 3.6. Resolución  de  consejo  directivo  organismo  supervisor  de  la  inversión  en  energía OSINERGMIN N° 028-2003-OS/CD (A-38)
- 3.7. Artículos 185°, 186° y 283° del Código Penal.

## 4. RECURSOS HUMANOS Y RESPONSABILIDADES

- 4.1. El  Comité  del  Sistema  Integrado de Gestión (CSIG) es  responsable  de  aprobar  el presente instructivo.
- 4.2. El  personal  de  Oficina  de  control  de  Pérdidas, son los responsables de aplicar lo establecido en el presente instructivo.

## 5. INFRAESTRUCTURA, RECURSOS Y AMBIENTE

- 5.1. SIELSE - Información de clientes.
- 5.2. Oficinas / Instalaciones / Zona de concesión de Electro Sur Este S.A.A.

## 6. DESARROLLO / PROCEDIMIENTO

El presente instructivo se encuentra desarrollado en:

- 6.1. Flujo N°1 - Análisis de información de clientes comunes.
- 6.2. Flujo N°2 - Análisis de información de clientes mayores.

## 7. REGISTROS Y ANEXOS

Los registros generados son:

- 7.1. Listado de clientes observados

Código

Versión

Aprobado

Fecha

Página

: CVPC-IN-002

: 00

: CSIG

: 28-06-2018

: 2 de 8

<!-- image -->

Fecha

Página

: CVPC-IN-002

: 00

: CSIG

: 28-06-2018

: 3 de 8

## INSTRUCTIVO

## ANÁLISIS DE INFORMACIÓN DE CLIENTES

## FLUJO N°1 - ANÁLISIS DE INFORMACIÓN DE CLIENTES COMUNES

<!-- image -->

<!-- image -->

Código

Versión

Aprobado

Fecha

Página

: CVPC-IN-002

: 00

: CSIG

: 28-06-2018

: 4 de 8

## INSTRUCTIVO

## ANÁLISIS DE INFORMACIÓN DE CLIENTES

| Proveedo- res                      | Entradas                 |   N° | Descripción Actividades                                                                                                                                                                                                                                                                 | Salidas                  | Clientes                           | Ejecutor de la Actividad           | Sistema que soporta la Actividad   | ¿Es control Cable? (Sí=x)   |
|------------------------------------|--------------------------|------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|------------------------------------|------------------------------------|------------------------------------|-----------------------------|
| -                                  | -                        |    1 | Solicitar información de clientes comunes a centro de servicio TI SIELSE - Consumos históricos - Lecturas Observadas - Suministros deudores y otros                                                                                                                                     | Solicitud de información | Centro de servi- cio TI SIELSE     | Supervisor de pérdidas Comerciales | Correo Electrónico                 |                             |
| Supervisor de pérdidas Comerciales | Solicitud de información |    2 | Enviar información solicitada                                                                                                                                                                                                                                                           | Datos                    | Supervisor de pérdidas Comerciales | Centro de servi- cio TI SIELSE     | Correo Electrónico                 |                             |
| Centro de servicio TI SIELSE       | Datos                    |    3 | Analizar evolución de consumos, Lecturas observadas y deudores - Baja de evolución de consumos Ejemplo: Mes1, Mes2, Mes3, Mes4, ….., Mes8, Mes9, Mes10 Análisis 1: Análisis 2: - Consumos cero - Características del medidor - Tiempo y/o monto de deudas - Lecturas observadas y otros | Listado de clientes      | Supervisor de pérdidas Comerciales | Supervisor de pérdidas Comerciales | SIELSE                             |                             |
| Supervisor de pérdidas Comerciales | Listado de clientes      |    4 | ¿Hay Observados? Si: Va a actividad 08 No: Va a actividad 07                                                                                                                                                                                                                            | -                        | Supervisor de pérdidas Comerciales | Supervisor de pérdidas Comerciales | No Aplica                          |                             |

<!-- image -->

Código

Versión

Aprobado

Fecha

Página

: CVPC-IN-002

: 00

: CSIG

: 28-06-2018

: 5 de 8

## INSTRUCTIVO

## ANÁLISIS DE INFORMACIÓN DE CLIENTES

| Proveedo- res                      | Entradas            |   N° | Descripción Actividades                                                                                                                                                                  | Salidas                        | Clientes                                | Ejecutor de la Actividad           | Sistema que soporta la Actividad   | ¿Es control Cable? (Sí=x)   |
|------------------------------------|---------------------|------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------|-----------------------------------------|------------------------------------|------------------------------------|-----------------------------|
| Supervisor de pérdidas Comerciales | -                   |    5 | Se retiran a los clientes de la lista de clientes siendo analizados                                                                                                                      | Listado de clientes            | Supervisor de pérdidas Comerciales      | Supervisor de pérdidas Comerciales | SIELSE                             |                             |
| Supervisor de pérdidas Comerciales | Listado de clientes |    6 | ¿Cliente ya fue visitado? Si: Va a actividad 07 No: Va a actividad 08                                                                                                                    | -                              | Supervisor de pérdidas Comerciales      | Supervisor de pérdidas Comerciales | No Aplica                          |                             |
| Supervisor de pérdidas Comerciales | -                   |    7 | ¿Fue hurto? Si: Va a actividad 08 No: Va a actividad 05                                                                                                                                  | -                              | Supervisor de pérdidas Comerciales      | Supervisor de pérdidas Comerciales | No Aplica                          |                             |
| Supervisor de pérdidas comerciales | -                   |    8 | Agrupar suministros a intervenir Por parámetros: - Cantidades / zonas / Tipos de zonas                                                                                                   | Listado de clientes            | Supervisor de pérdidas Comerciales      | Supervisor de pérdidas Comerciales | SIELSE                             |                             |
| Supervisor de pérdidas Comerciales | Listado de clientes |    9 | ¿Corresponde priorizar suministro? Si: Va a actividad 10 No: Va a actividad 11                                                                                                           | -                              | Supervisor de pérdidas Comerciales      | Supervisor de pérdidas Comerciales | No Aplica                          |                             |
| Supervisor de pérdidas Comerciales | -                   |   10 | Aplicar criterios de priorización - Zonas críticas en pérdidas. - Consumos importantes.                                                                                                  | Listado de clientes            | Supervisor de pérdidas Comerciales      | Supervisor de pérdidas Comerciales | SIELSE                             |                             |
| Supervisor de pérdidas Comerciales | Listado de clientes |   11 | Verificar cantidades por zonas Para ver si exceden el límite establecido: - Metas trimestrales del POE. - Contratos de los servicios tercerizados.                                       | Listado de clientes            | Supervisor de pérdidas comerciales      | Supervisor de pérdidas Comerciales | SIELSE                             |                             |
| Supervisor de pérdidas Comerciales | Listado de clientes |   12 | ¿Es adecuada la cantidad? Si: Va a actividad 13 No: Va a actividad 03 (Añadir parámetros y/o criterios)                                                                                  | -                              | Supervisor de pérdidas Comerciales      | Supervisor de pérdidas Comerciales | No Aplica                          |                             |
| Supervisor de pérdidas comerciales | -                   |   13 | Cerrar listado y adjuntar datos necesarios para la inter- vención Para su atención según CVPC-PR-001 Verificación de su- ministro, detección de vulneración y/o consumo no autoriza- do. | Listado de clientes observados | Grupo de tra- bajo ELSE y/o Contratista | Supervisor de pérdidas Comerciales | Correo Electrónico                 |                             |

<!-- image -->

Fecha

Página

: CVPC-IN-002

: 00

: CSIG

: 28-06-2018

: 6 de 8

## INSTRUCTIVO

## ANÁLISIS DE INFORMACIÓN DE CLIENTES

## FLUJO N°2 - ANÁLISIS DE INFORMACIÓN DE CLIENTES MAYORES

<!-- image -->

<!-- image -->

Código

Versión

Aprobado

Fecha

Página

: CVPC-IN-002

: 00

: CSIG

: 28-06-2018

: 7 de 8

## INSTRUCTIVO

## ANÁLISIS DE INFORMACIÓN DE CLIENTES

| Proveedo- res                      | Entradas                     |   N° | Descripción Actividades                                                                                                                                                                                                                             | Salidas                             | Clientes                           | Ejecutor de la Actividad           | Sistema que soporta la Actividad   | ¿Es control Cable? (Sí=x)   |
|------------------------------------|------------------------------|------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------|------------------------------------|------------------------------------|------------------------------------|-----------------------------|
| -                                  | -                            |    1 | Solicitar información de clientes comunes a centro de servicio TI SIELSE - Histórico de consumos - Histórico de factor de transformación de facturación - Factor de transformación actual. - Perfiles de carga. - Potencia de transformador y otros | Solicitud de in- formación          | Supervisor de pérdidas comerciales | Supervisor de pérdidas comerciales | Correo Electrónico                 |                             |
| Supervisor de pérdidas Comerciales | Solicitud de infor- mación   |    2 | Enviar información solicitada                                                                                                                                                                                                                       | Datos                               | Supervisor de pérdidas Comerciales | Centro de servicio TI SIELSE       | Correo Electrónico                 |                             |
| Centro de servicio TI SIELSE       | Datos                        |    3 | Evaluar factores de transformación Si los datos recogidos en campo son correctos se actualizan datos en fichero                                                                                                                                     | Factores de trans- formación        | Supervisor de pérdidas Comerciales | Supervisor de pérdidas comerciales | SIELSE                             |                             |
| Supervisor de pérdidas Comerciales | -                            |    4 | Informar posibles factores de transformación Los datos son recogidos en campo                                                                                                                                                                       | Posibles factores de transformación | Supervisor de pérdidas Comerciales | Supervisor de pérdidas comerciales | SIELSE                             |                             |
| Supervisor de pérdidas Comerciales | Factores de trans- formación |    5 | ¿Es Correcto? Si: Va a actividad 06 No: Va a actividad 12                                                                                                                                                                                           | -                                   | Supervisor de pérdidas Comerciales | Supervisor de pérdidas comerciales | No Aplica                          |                             |
| Supervisor de pérdidas Comerciales | -                            |    6 | Analizar evolución de consumos Ejemplo: Mes1, Mes2, Mes3, Mes4, ….., Mes8, Mes9, Mes10 Análisis 1: Análisis 2:                                                                                                                                      | Datos de evolu- ción de consumo     | Supervisor de pérdidas Comerciales | Supervisor de pérdidas comerciales | No Aplica                          |                             |

<!-- image -->

Código

Versión

Aprobado

Fecha

Página

: CVPC-IN-002

: 00

: CSIG

: 28-06-2018

: 8 de 8

## INSTRUCTIVO

## ANÁLISIS DE INFORMACIÓN DE CLIENTES

| Proveedo- res                      | Entradas                      |   N° | Descripción Actividades                                                                                                                                     | Salidas                          | Clientes                              | Ejecutor de la Actividad           | Sistema que soporta la Actividad   | ¿Es control Cable? (Sí=x)   |
|------------------------------------|-------------------------------|------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------|---------------------------------------|------------------------------------|------------------------------------|-----------------------------|
| Supervisor de pérdidas Comerciales | Datos de evolución de consumo |    7 | ¿Variación considerable? Si: Va a actividad 12 No: Va a actividad 08                                                                                        | -                                | Supervisor de pérdidas Comerciales    | Supervisor de pérdidas comerciales | No Aplica                          |                             |
| Supervisor de pérdidas Comerciales | -                             |    8 | Analizar los perfiles de carga                                                                                                                              | Perfiles de carga                | Supervisor de pérdidas Comerciales    | Supervisor de pérdidas comerciales | No Aplica                          |                             |
| Supervisor de pérdidas Comerciales | Perfiles de carga             |    9 | ¿Es Correcto? Si: Va a actividad 10 No: Va a actividad 12                                                                                                   | -                                | Supervisor de pérdidas Comerciales    | Supervisor de pérdidas comerciales | No Aplica                          |                             |
| Supervisor de pérdidas Comerciales | -                             |   10 | Analizar potencia de transformador                                                                                                                          | Potencia de trans- formador      | Supervisor de pérdidas Comerciales    | Supervisor de pérdidas comerciales | No Aplica                          |                             |
| Supervisor de pérdidas Comerciales | Potencia de trans- formador   |   11 | ¿Es Correcto? Si: Fin del proceso No: Va a actividad 12                                                                                                     | -                                | Supervisor de pérdidas Comerciales    | Supervisor de pérdidas comerciales | No Aplica                          |                             |
| Supervisor de pérdidas Comerciales | Clientes Observados           |   12 | Enviar listado a grupo de trabajo ELSE Para su atención según CVPC-PR-001 Verificación de suministro, detección de vulneración y/o consu- mo no autorizado. | Listado de clien- tes observados | Grupo de trabajo ELSE y/o Contratista | Supervisor de pérdidas Comerciales | Correo Electrónico                 |                             |

C

Copia