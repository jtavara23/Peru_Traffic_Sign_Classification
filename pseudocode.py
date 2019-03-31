
function funcionRELU(datos):
    Para cada i en datos:
        nuevosDatos[i] <- max(datos[i],0);
    retorna nuevosDatos;

function capaConvolucion(imagenes, numCanales,tamañoFiltro, numFiltros, usarPooling)	:
    forma <- [tamañoFiltro, tamañoFiltro, numCanales, numFiltros]
    pesos <- inicializar_pesos(shape=forma);
    biases <- inicializar_bias([numFiltros]);
    convolucion <-convolucion2D(input = imagenes, filter=pesos, formaPadding=[1, 1, 1, 1], padding=activado);

    convolucion <- convolucion + biases;
    
    convolucion <- funcionRELU(convolucion);

    if usarPooling:
        convolucionPool <- ejecutarPooling(convolucion,2)
    retorna (convolucionPool, pesos, convolucion)


function entropiaCruzada(vector_salida_calculada,vector_salida_deseada):
    cross_entropy <- -.sum(vector_salida_deseada * nlog(vector_salida_calculada));
    retorna cross_entropy;

function softmax(vector_datos):
    retorna exp(vector_datos)/sum(exp(vector_datos));


function capa_Totalmente_Conectada(entradas, num_Entradas, num_Salidas, usarRELU):
    
    pesos <- inicializar_pesos(forma=[num_Entradas, num_Salidas])
    biases <- inicializar_bias([num_Salidas])
    capa <- (entrada * pesos) + biases

    if usarRELU:
        capa <- funcionRELU(capa)
    else:

    return (capa, pesos)


main():
    entradas <-  TensorPlaceHolder(Lote_imagenes, tamañoImagen, tamañoImagen, Profundidad);
    salida_deseada <- TensorPlaceHolder(Lote_imagenes,NumeroClases);

    num_filtros_entrada_Capa_N <- 1;

    Para N <- Numero_De_Capas:
        capa_conv_N, pesos_conv_N, capa_conv_NoPool_N <- capaConvolucion(
            imagen=entradas,
            numCanales=num_filtros_entrada_Capa_N,
            tamañoFiltro=tam_filtro_Capa_N,
            numFiltros=num_filtros_salida_Capa_N,
            usarPooling=True)

        capa_conv_N_dropOut <- dropout(capa_conv_N, keep_prob=dropout_convN);

        entradas <- capa_conv_N_dropOut;                             
        num_filtros_entrada_Capa_N = num_filtros_salida_Capa_N;
    Fin

    Para N <- (Numero_De_Capas - 1):
        capa_conv_N_dropOut <- ejecutarPooling(data=capa_conv_N_dropOut, tamañoFiltro= 2^(Numero_De_Capas-N));
        datos_capa_N <- datos_capa_N + capa_conv_N_dropOut[ancho] * capa_conv_N_dropOut[largo] * capa_conv_N_dropOut[canales];
        capa_Salida <- capa_Salida + capa_conv_N_dropOut;
    Fin


    capa_fc1, pesos_fc1 <- capa_Totalmente_Conectada(
            imagenes=capa_Salida,
            num_Entradas=datos_capa_N,
            num_Salidas=num_Salidas,
            usarRELU=True);

    capa_fc1_dropOut <- dropout(capa_fc1, keep_prob=dropout_fc1);


    capa_fc2, pesos_fc2 <- capa_Totalmente_Conectada(
            imagenes=capa_fc1_dropOut,
            num_Entradas=num_Salidas,
            num_Salidas=num_Clases,
            usarRELU=False);

    salida_calculada <- softmax(capa_fc2);

    error <- entropiaCruzada(salida_calculada,salida_deseada);
    optimizador = OptimizadorAdam(Tasa_Aprendizaje).minimize(error, global_step=iterac_entren)