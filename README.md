### Formulación general del problema de clasificación

- [ ] Reservar un subconjunto de ejemplos para Test y usar el resto para entrenar.
- [ ] Preprocesar los ejemplos (si es necesario)
- [ ] Hacer una representación One-hot de la variable target (si es necesario)
- [ ] Elegir una función de pérdida adecuada para la tarea y los datos
- [ ] Entrenar un modelo con las decisiones previas
- [ ] Evaluar el rendimiento de ese modelo.
- [ ] Repetir para mejorar los resultados

### Apuntes clasificador multiclase

- En problemas multiclase con representacion one-hot la red tendrá tantas neuronas de salida como clases haya.
- Cada neurona de salida produce un valor antes de la activación, logit, estos pueden tomar cualquier valor, positivo o negativo.
- Sea z_i el logit de la neurona i-ésima, la función Softmax se define como:
$$ \text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}} = \hat{p}(y = clase \ i) $$
- Utilizando la función Softmax como función de activación para todas las neuronas de salida, la salida de la red es la distribución de probabilidades sobre las M clases.
- Utilizaremos la Cross-Entropy como función de pérdida, que para una muestra x_i y su etiqueta y_i es:
$$ CE(P_{true}(y), P(\hat{y}|\mathbf{X},\mathsf{w})) \simeq - \frac{1}{N} \sum_{i=1}^N P_{true}(y^{(i)}) \cdot \log \hat{p}(y = clase \ i) $$
