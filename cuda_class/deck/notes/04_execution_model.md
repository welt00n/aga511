## divisor
• Parte 3, ~16 min, a maior dos fundamentos
• guia 1.2, 2.1, 2.3, com o nosso código
• tudo o que o Buffon precisa
Parte 3: o primeiro capítulo do guia da NVIDIA, o modelo de execução. Vou usar as seções 1.2, 2.1 e 2.3 do guia, mas sempre com o nosso código como exemplo.
Tudo o que o Buffon precisa está aqui: como se lança um kernel, o que são blocos e warps, como os threads cooperam numa redução, e os atômicos.

## Host e device
• dois computadores, duas memórias
• cudaMalloc, cudaMemcpy (PCIe), lançamento volta na hora
• __global__, __device__, __host__ __device__
A primeira coisa que muda em relação ao C: são dois computadores, com duas memórias. O host é a CPU com a RAM; o device é a GPU com a memória dela. Um ponteiro da GPU não vale na CPU, e vice-versa: os dados viajam pelo barramento PCIe, de propósito.
Três chamadas fazem quase tudo: cudaMalloc reserva memória na GPU; cudaMemcpy copia; e o lançamento do kernel, com os três sinais de menor, volta na hora, e a CPU segue.
E três marcas nas funções: __global__ é o kernel, roda na GPU e é chamado pela CPU; __device__ só roda na GPU; e __host__ __device__ compila para os dois. É assim que a física do EP4 é uma função só, a mesma na CPU e na GPU.
▶ Clique: as três linhas da API.

## Um kernel mínimo
• soma de vetores: o exemplo do guia
• i = blockIdx.x · blockDim.x + threadIdx.x
• if (i < n): o último bloco passa do fim
• desenho: bloco 2, thread 5 → i = 21
O primeiro exemplo de todo curso de CUDA, e do guia: somar dois vetores. Cada thread executa exatamente o mesmo código. A única coisa que muda é o índice: blockIdx vezes blockDim mais threadIdx, que é "quem sou eu". O guia chama essa conta de muito comum, e é verdade: está em quase todo kernel.
O if existe porque o último bloco pode passar do fim do vetor.
No desenho, com blocos de 8 threads: o thread pintado é o 5 do bloco 2, então ele calcula o elemento 2 vezes 8 mais 5, o 21.
▶ Clique: destaca o índice. Clique: o lançamento.
▶ Perguntar: "com n = 1000 e blocos de 256, quantos blocos?" (4)

## O programa inteiro: o lado do host
• 5 passos: reservar, copiar, lançar, copiar de volta, liberar
• lançamento assíncrono; a cópia de volta espera
• EP4: o mesmo, com check(...) em tudo
Este é o lado do host, o programa inteiro: reservar, copiar para lá, lançar, copiar de volta, liberar. Cinco passos, sempre os mesmos.
Um detalhe que o guia faz questão de dizer: o lançamento é assíncrono, ele volta antes de o kernel terminar. Quem espera é a cópia de volta, que é síncrona.
O EP4 tem exatamente esse esqueleto, com uma verificação de erro em volta de cada chamada, o que o guia chama de boa prática.
▶ Cada clique destaca um passo.

## Thread, warp, bloco, grid, e os SMs
• thread → warp (32) → bloco (≤ 1024) → grid → SMs
• o nosso FP64: blocos de 768 threads, 36 blocos, 1 por SM
• pergunta: 768 threads = quantos warps? (24)
A hierarquia, de baixo para cima. O thread executa o kernel uma vez. O warp são 32 threads que andam juntos, executando a mesma instrução. O bloco é um grupo de até 1024 threads que podem cooperar, e vai inteiro para um SM. E o grid são todos os blocos de um lançamento, que se distribuem pelos SMs.
O nosso caso em FP64: blocos de 768 threads, 36 blocos, um em cada SM.
Pergunta: um bloco de 768 threads tem quantos warps?
▶ Um clique por nível: warp, bloco, grid, SMs.
▶ Clique: a pergunta. Resposta: 24.

## O fluxo de execução
• copiar → lançar → executar → copiar de volta
• antes: o contexto CUDA, ~0,15 s (EP3)
• pergunta: onde se esconde a parte serial?
O fluxo de um programa CUDA: copiar os dados para a GPU, lançar, os SMs trabalham, copiar de volta. E antes de tudo, criar o contexto CUDA: no EP3, medi com o Nsight Systems, uns 0,15 segundo.
Calcular é barato; mover dados não é. Guardem isso para quando chegarmos em Amdahl.
Pergunta: onde vocês esperariam que a parte serial de um programa de GPU se escondesse?
▶ Um clique por seta.
▶ Clique: a pergunta. Deixar responder. No EP3 é o contexto; no EP4 vai ser a imagem.

## SPMD
• Single Program, Multiple Data: um programa, escrito para UM thread
• lançado como 27 648 instâncias, cada uma com os seus dados
• o mesmo estilo do MPI: lá processos, aqui threads
• SPMD = como escrevemos
Antes de falar de como a GPU executa, quero separar duas ideias que costumam se misturar.
A primeira é SPMD: Single Program, Multiple Data, um programa, muitos dados. O nosso kernel, trace_photons, está escrito para um thread só: pega um pedaço da fila, traça os fótons dele, soma os resultados. E ele é lançado como 27 648 instâncias, cada uma com os seus pedaços e o seu caminho pelo código.
É o mesmo estilo de um programa MPI, que é o tema de outro grupo: lá, o mesmo programa roda em vários processos, cada um com o seu rank; aqui, em vários threads, cada um com o seu índice. O termo é dos anos 80, da IBM.
SPMD é o modelo de programação: diz como pensamos e escrevemos. Não diz nada sobre como o hardware executa.
▶ Clique: a conclusão. Dizer: e agora, como a GPU executa isso?

## SIMT: a GPU executa em warps
• SIMD (CPU): registrador de largura fixa; o programa vetoriza
• SIMT (GPU): o hardware agrupa 32 threads e emite uma instrução
• cada thread com os seus registradores e o seu caminho
• guia: SIMT não tem largura fixa como SIMD
A segunda ideia é SIMT, como a GPU executa. Para entender, comparem com o SIMD da CPU, à esquerda: uma instrução AVX soma quatro doubles de uma vez, num registrador de largura fixa. Quem vetoriza é o programador ou o compilador, e a largura faz parte do programa. Um if vira uma máscara explícita.
À direita, a GPU. O programa foi escrito para um thread, em SPMD. É o hardware que agrupa 32 threads num warp e emite uma instrução para os 32. Cada thread tem os seus próprios registradores e, desde 2017, o seu próprio contador de programa: pode seguir o seu caminho.
O guia diz assim: SIMT não tem uma largura de dados fixa como SIMD. Resumindo: SPMD é como escrevemos; SIMT é como a GPU executa. E quando os threads de um warp querem caminhos diferentes, aparece a divergência.
▶ Clique: a conclusão.

## A divergência
• if que divide o warp: um lado, depois o outro
• o warp paga a() + b()
• EP4: fóton bloqueado por mancha termina antes, e espera
O que acontece quando os 32 threads de um warp não concordam num if? A GPU executa um lado, com os threads do outro lado parados, e depois o outro lado. O warp gasta o tempo de a() mais o de b(). Isso é divergência.
No EP4 isso acontece de verdade: um fóton que bate numa mancha termina antes, e a faixa dele fica esperando o resto do warp.
▶ Cliques: a(), b(), depois do if. Clique: a frase.

## Redução paralela
• somar N números
• distribuição → parciais → árvore → resultado final
• serial N passos; paralelo N/p + log₂ p
• metade dos threads para a cada nível
O exemplo prático que o roteiro do professor pede: a redução paralela, somar N números.
Distribuição: cada thread pega o seu número. Combinação em árvore: a cada passo, metade dos threads soma o valor de um vizinho. No fim, um thread tem a soma do bloco, e faz um atômico no total.
Serialmente são N passos. Em paralelo, N sobre p mais log de p. E reparem no cinza: a cada nível metade dos threads para. É a limitação da redução, e o motivo de a árvore ser curta.
▶ Um clique por nível. No fim, o atomicAdd.

## A redução em CUDA
• memória compartilhada (do bloco)
• __syncthreads() por nível
• só sincroniza dentro do bloco; ~100 KB por SM
O mesmo desenho, em código. O vetor parcial está na memória compartilhada, que é do bloco. Depois de cada nível, um __syncthreads, para que todos terminem antes do próximo passo.
Duas limitações: __syncthreads só sincroniza dentro de um bloco; entre blocos, o guia é explícito, não há mecanismo. E a memória compartilhada é pequena, uns 100 KB por SM.
▶ Cliques: compartilhada, distribuição, árvore, atômico.

## Quando todos escrevem no mesmo lugar
• corrida: os dois leem 5, os dois escrevem 6
• atomicAdd: ler-somar-escrever indivisível, na L2
• Buffon: 1 atômico por thread; estrela: 1 por fóton, em bilhões de pixels
O que acontece quando vários threads escrevem no mesmo lugar? À esquerda, sem atômico: os dois threads leem 5, os dois somam 1, os dois escrevem 6. Uma soma se perdeu. É uma condição de corrida.
O atomicAdd faz ler, somar e escrever como uma operação indivisível, na cache L2. No mesmo endereço, os atômicos fazem fila; em endereços diferentes, andam em paralelo.
O Buffon soma em registrador e faz um atômico só por thread. A estrela faz um por fóton, mas em bilhões de pixels diferentes, então quase nunca há fila.
▶ Clique: o atomicAdd. Clique: a nota.

## E se houver mais elementos que threads?
• grid-stride loop: passo = blockDim.x · gridDim.x
• divisão fixa: o thread k faz k, k + passo, …
• EP3/EP4: divisão dinâmica, uma fila
O kernel mínimo lança um thread por elemento. Mas e se houver mais elementos que threads? O padrão, que o guia mostra, é o grid-stride loop: um grid do tamanho da GPU, e cada thread anda de passo em passo, onde o passo é o número total de threads.
Aqui a divisão é fixa: o thread k faz o elemento k, depois k mais o passo, e assim por diante.
No EP3 e no EP4 a divisão é dinâmica: uma fila. Quem termina antes pega mais. Isso importa quando cada elemento custa um tempo diferente, como um fóton que bate numa mancha.
▶ Cliques: o passo, o laço. Clique: grid-stride. Clique: a fila.

## Os blocos vão inteiros
• guia 1.2.2.1: um bloco, um SM; qualquer ordem
• inteiro: __syncthreads precisa de todos vivos
• 25 warps cabem, lançamos 24
• pergunta: bloco 0 escreve uma flag, bloco 1 espera. Funciona?
O guia é bem claro: todos os threads de um bloco rodam num único SM, os blocos rodam em qualquer ordem, e não há garantia de escalonamento entre eles.
Por que o bloco vai inteiro? Porque os threads dele podem chamar __syncthreads, e para isso precisam estar todos vivos ao mesmo tempo. Por isso, com 80 registradores, cabem 25 warps no SM, mas nós lançamos 24: um bloco de 24 warps, inteiro. A conta vem na parte 8.
Uma pergunta: o bloco 0 escreve uma flag, e o bloco 1 fica esperando por ela. Funciona?
▶ Cliques: inteiro, 25 e 24, qualquer ordem. Clique: a pergunta.
▶ Resposta: depende da ordem; se o bloco 1 ocupar o lugar e ficar esperando, o 0 nunca entra e a GPU trava. Nos EPs, os blocos só se falam por atômicos que nunca esperam.

## Rodando o nosso kernel
• 36 blocos × 768 threads, 1 por SM, todos juntos
• --sms 12: os outros 24 leem %smid e voltam
• os 12 pegam toda a fila → Amdahl em 1…36 SMs
Agora o nosso kernel rodando. Em FP64: 36 blocos de 768 threads, um em cada SM, todos ao mesmo tempo.
E aqui está o truque que me deixa medir Amdahl. Com a opção --sms 12, cada bloco lê o número do seu SM, um registrador especial chamado %smid. Os blocos que estão fora dos 12 escolhidos voltam na hora. Os 12 que ficam pegam todos os pedaços da fila.
É o mesmo programa, só muda quantos SMs trabalham. É assim que medi 1, 2, 3, até 36 SMs.
▶ Um clique por item.

## Threads persistentes
• API de ocupação → lançar exatamente o que cabe
• CS149: "persistent threads"; o risco: supor todos juntos
• lançamento cooperativo: todos entram, ou falha (guia 4.4.8)
• uma leva só → a máscara mede de fato n SMs
Este é o código real do lançamento. A API de ocupação do CUDA diz quantos threads cabem por bloco e quantos blocos por SM; e eu lanço exatamente o que cabe.
Em Stanford, no curso CS149, isso se chama persistent threads. E eles apontam o risco: o código supõe que todos os blocos rodam ao mesmo tempo, e CUDA não promete isso.
A nossa garantia é o lançamento cooperativo. O guia diz que ele é atômico: se a chamada funciona, todos os blocos entram juntos; se não cabem, ela falha. Com uma leva só, cada SM recebe exatamente os seus blocos, e a máscara mede de verdade n SMs.
▶ Cliques no código: ocupação, lançamento, máscara, fila. Depois, um por item.
