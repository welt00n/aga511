## divisor
• Parte 7, ~8 min
• guia 2.3.3: onde mora cada dado
Parte 7: o terceiro capítulo do guia, a memória. Onde mora cada dado, e o que acontece quando todos os threads escrevem na mesma imagem.

## O modelo de memória
• perto do thread: rápida e pequena
• registradores (thread), compartilhada (bloco), global (todos), constante (todos, só leitura)
• EP4: fóton, estrela, imagem
A regra geral: quanto mais perto do thread, mais rápida e menor. Registradores são de um thread. A memória compartilhada é de um bloco. A global é de todos, tem gigabytes, e custa centenas de ciclos. A constante é de todos, só para leitura.
▶ Clique: o EP4. O fóton vive em registradores, a estrela em memória constante, a imagem em memória global.

## A tabela do guia, com uma coluna nossa
• fóton: 80 registradores (FP64)
• local: 40 bytes de pilha, 0 derramados
• compartilhada: não usamos
• global: imagem de 5 GB; constante: estrela de 18 KB
Esta é a tabela 1 do guia, com uma coluna a mais: o que o EP4 guarda em cada espaço. Os números vêm do orquestrador.
O fóton ocupa 80 registradores por thread, em FP64. Na memória local, 40 bytes de pilha, nada derramado. A memória compartilhada não usamos: a imagem não cabe. Na global, a imagem de 5 gigabytes, a fila e as somas. E na constante, a descrição da estrela: 18 mil bytes.
▶ Clique: "local" é o escopo, não o lugar: fica na memória do device, com a latência da global.
▶ Clique: é para lá que vão os derramamentos da parte 8.

## Constante ou global?
• constante: uma leitura serve o warp inteiro
• medido, estrela em global: ciclos 0,996× (FP64), 1,007× (FP32)
• ruído; a estrela cabe na cache
• mantemos a constante: diz o que o dado é
O guia sugere a memória constante quando todos os threads leem o mesmo endereço: uma leitura serve o warp inteiro. É exatamente o caso da estrela.
Mas eu testei: compilei uma versão com a estrela na memória global. Resultado: 0,996 vezes os ciclos em FP64 e 1,007 em FP32. Dentro do ruído. A estrela tem 18 KB, cabe nas caches, e a L1 também entrega o mesmo dado a todos.
Mantenho a constante porque ela diz o que o dado é: só leitura, igual para todos. E a versão em global ainda pede mais registradores.
▶ Clique: a conclusão. Dizer: predizer e medir; às vezes a recomendação não muda nada.

## A imagem é um histograma
• guia: histograma na compartilhada, depois atômicos na global
• o nosso: 1,26·10⁹ contagens = 5 GB → direto na global
• bilhões de endereços: quase nunca fila
• contagens inteiras: mesma soma em qualquer ordem → bit a bit
A imagem da estrela é um histograma de fótons: cada fóton soma 1 num pixel. O guia ensina como fazer histogramas: cada bloco conta na sua memória compartilhada, e no fim soma tudo na global com atômicos.
O nosso histograma tem 1024 ao quadrado pixels, vezes 300 quadros, vezes 4 observadores: 1,26 bilhão de contagens, 5 gigabytes. Não cabe em 100 KB. Então é um atomicAdd por fóton, direto na memória global. Com bilhões de endereços, dois fótons quase nunca disputam o mesmo pixel.
E o ponto mais bonito: as contagens são inteiras. Somar inteiros em qualquer ordem dá o mesmo número. Com ponto flutuante, o guia avisa, a ordem dos atômicos muda o resultado.
▶ Clique: a conclusão. Dizer: é uma das razões de a imagem sair bit a bit.
