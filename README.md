# rut5_siam
Пример сиамской модели на основе https://gist.github.com/avidale/dc7a26eb3cffc90075bf100e15b5950f
В качестве головы используется мультилинейный сиамский слой. Например
    MODEL_DIM = 512
    head_sizes=[MODEL_DIM] * 2,
    classifier_sizes=[MODEL_DIM * 2,
        MODEL_DIM,
        1]
Голова - это слой над эмбедером, а классификатор - объедияняющий слой для двух голов.
