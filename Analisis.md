# Análisis de la Evaluación de Mass Sculpting

Este gráfico presenta los resultados de validación del clasificador tras ajustar la penalidad de decorrelación. El comportamiento observado es el ideal para análisis en Física de Altas Energías (HEP).

## Observaciones Técnicas

### 1. Preservación de la Cinemática del Fondo
La característica más destacable es que las cuatro distribuciones se **superponen casi a la perfección**. Al aplicar cortes de selección más estrictos (desde el 100% hasta el 1% del fondo), la forma macroscópica de la distribución de masa no se altera. Esto confirma que el clasificador es **agnóstico a la masa invariante**, evitando sesgos en la selección.

### 2. Ausencia de "Mass Sculpting"
Al observar la línea roja (top 1% de eventos con mayor score), se aprecia que se mantiene la **caída exponencial suave** típica del fondo QCD. 
* No hay formación de picos espurios (falsas resonancias).
* No se observan "jorobas" artificiales en las regiones de interés (como los rangos de 80-90 GeV o 125 GeV).
* Se valida que el modelo no utiliza la masa como una variable discriminante indirecta.

### 3. Fluctuaciones Estadísticas
La curva roja (corte al 1%) presenta un perfil más "dentado" que la curva original del 100%. Este es el **comportamiento estadístico esperado** debido a la reducción del tamaño de la muestra en cada *bin*. Estas variaciones son fluctuaciones de Poisson estándar y no representan un esculpido físico de la distribución.

---

## Conclusión sobre la Decorrelación

El ajuste en la magnitud de la penalidad **DisCo** ha sido exitoso. La arquitectura (VAE/GNN) ha logrado extraer características de la subestructura del jet en un espacio latente que es efectivamente independiente de la masa. 

Este nivel de decorrelación asegura que cualquier señal observada tras la aplicación del clasificador en datos reales será atribuible a la **física subyacente** y no a un artefacto del modelo de aprendizaje profundo. El pipeline de inferencia es robusto y está listo para la fase de análisis de sensibilidad.
