
function funcionRELU(datos):
    Para cada i en datos:
        nuevosDatos[i] <- max(datos[i],0);
    retorna nuevosDatos;

function capaConvolucion(imagenes, numCanales,tamañoFiltro, numFiltros, usarPooling)	:
    forma <- [tamañoFiltro, tamañoFiltro, numCanales, numFiltros]
    pesos <- inicializarpesos(shape=forma);
    biases <- inicializarbias([numFiltros]);
    convolucion <-convolucion2D(input = imagenes, filter=pesos, formaPadding=[1, 1, 1, 1], padding=activado);

    convolucion <- convolucion + biases;
    
    convolucion <- funcionRELU(convolucion);

    if usarPooling:
        convolucionPool <- ejecutarPooling(convolucion,2)
    retorna (convolucionPool, pesos, convolucion)


function entropiaCruzada(vectorsalidacalculada,vectorsalidadeseada):
    crossentropy <- -.sum(vectorsalidadeseada * nlog(vectorsalidacalculada));
    retorna crossentropy;

function softmax(vectordatos):
    retorna exp(vectordatos)/sum(exp(vectordatos));


function capaTotalmenteConectada(entradas, numEntradas, numSalidas, usarRELU):
    
    pesos <- inicializarpesos(forma=[numEntradas, numSalidas])
    biases <- inicializarbias([numSalidas])
    capa <- (entrada * pesos) + biases

    if usarRELU:
        capa <- funcionRELU(capa)
    return (capa, pesos)


main():
    entradas <-  TensorPlaceHolder(Loteimagenes, tamañoImagen, tamañoImagen, Profundidad);
    salidadeseada <- TensorPlaceHolder(Loteimagenes,_NumeroClases);

    numfiltrosentradaCapa_N <- 1;

    Para _N <- _NumeroDeCapas:
        capaconv_N, pesosconv_N, capaconv_NoPool_N <- capaConvolucion(
            imagen=entradas,
            numCanales=numfiltrosentradaCapa_N,
            tamañoFiltro=tamfiltroCapa_N,
            numFiltros=numfiltrossalidaCapa_N,
            usarPooling=True)

        capaConvDropOut_N <- dropout(capaconv_N, keepprob=dropoutconv_N);

        entradas <- capaConvDropOut_N;                             
        numfiltrosentradaCapa_N = numfiltrossalidaCapa_N;
    Fin

    Para _N <- (_NumeroDeCapas - 1):
        capaConvDropOut_N <- ejecutarPooling(data=capaConvDropOut_N, tamañoFiltro= 2^(_NumeroDeCapas-_N));
        datosCapa_N <- datosCapa_N + capaConvDropOut_N[ancho] * capaConvDropOut_N[largo] * capaConvDropOut_N[canales];
        capaSalida <- capaSalida + capaConvDropOut_N;
    Fin


    capafc1, pesosfc1 <- capaTotalmenteConectada(
            imagenes=capaSalida,
            numEntradas=datosCapa_N,
            numSalidas=numSalidas,
            usarRELU=True);

    capafc1dropOut <- dropout(capafc1, keepprob=dropoutfc1);


    capafc2, pesosfc2 <- capaTotalmenteConectada(
            imagenes=capafc1dropOut,
            numEntradas=numSalidas,
            numSalidas=numClases,
            usarRELU=False);

    salidacalculada <- softmax(capafc2);

    error <- entropiaCruzada(salidacalculada,salidadeseada);
    optimizador = OptimizadorAdam(TasaAprendizaje).minimize(error, globalstep=iteracentren)