## divisor
• Parte 1, ~10 min
• cada peça da GPU responde a um problema
Parte 1: história. Não é uma lista de datas. A promessa é que cada peça de uma GPU existe porque alguém teve um problema, e a história explica a arquitetura.
A cada época, vou escrever no quadro o conceito que ela deixou.
▶ Preparar o quadro: uma linha do tempo.

## Ciência e indústria, juntas
• ciência: Cray-1 (1976), Connection Machine (1985)
• indústria: placas de vídeo, anos 90
• juntas desde 2007
O paralelismo nasceu na ciência. O Cray-1, de 1976, fazia uma operação num vetor inteiro de uma vez. A Connection Machine, de 1985, tinha 65 536 processadores simples, todos fazendo a mesma coisa.
Depois a ideia foi para a indústria, nas placas de vídeo dos anos 90, para desenhar jogos. E a partir de 2007, com CUDA, ciência e indústria andam juntas: o mesmo chip serve a jogos, a inteligência artificial e a supercomputadores.
Então não é bem que "a ciência descobriu a GPU". É uma volta: a ideia saiu da ciência, ficou barata na indústria, e voltou.
▶ Clique: a faixa "juntas, desde 2007".
▶ Clique: CUDA, a plataforma dos últimos 20 anos.

## 1999: milhões de pixels por quadro
• GeForce 256: cada pixel, a mesma conta, sem vizinhos
• pergunta: o que esse problema pede do hardware?
• vazão, não latência
Em 1999, a GeForce 256. O trabalho dela era desenhar milhões de pixels por quadro, sessenta vezes por segundo. E cada pixel é calculado com a mesma conta, sem esperar pelos vizinhos.
Pergunto para vocês: o que esse problema pede do hardware?
Pede muitas unidades simples, em vez de poucas rápidas. O que importa é quantas contas por segundo saem no total, a vazão, e não quanto tempo leva cada uma, a latência.
▶ Perguntar antes de revelar. Esperar uma ou duas respostas.
▶ Clique: o conceito. Quadro: paralelismo de dados; vazão.

## 2001–2006
• shaders: um programa por pixel = o kernel
• GPGPU: matrizes disfarçadas de texturas
• faltava um modelo de programação
Em 2001 a GeForce 3 trouxe os shaders programáveis: um pequeno programa que roda uma vez para cada pixel. Isso já é o que hoje chamamos de kernel: o mesmo programa, executado para cada elemento.
Os cientistas perceberam e começaram a pegar carona. Em 2004, o BrookGPU, de Stanford, escondia matrizes dentro de texturas e lia os resultados como se fossem imagens. Funcionava, mas era preciso fingir que tudo era gráfico.
▶ Clique: o conceito. Quadro: o kernel.

## 2006–2007: CUDA
• G80: núcleos iguais em multiprocessadores, os SMs
• CUDA: programar em C, sem fingir gráfico
• o SM: a unidade desta aula
Em 2006 a NVIDIA lançou a G80, que trocou os estágios fixos do pipeline gráfico por núcleos iguais, agrupados em multiprocessadores: os SMs. E introduziu CUDA, que saiu em 2007: programar a GPU em C, sem fingir que é gráfico.
Foi o maior salto de uma geração para a outra, porque o mesmo chip passou a servir para jogos e para ciência.
Guardem o SM: ele é a unidade desta aula inteira. Nos dois projetos, comparo um SM com todos.
▶ Clique: o conceito. Quadro: host, device, lançamento, SM.

## 2008–2010
• FP64 (2008), ECC e caches (Fermi, 2010)
• FP32: 24 bits; FP64: 53 bits
• unidades FP64 separadas: 2 no laptop, 32 na A30
A ciência pediu duas coisas: precisão dupla, o FP64, que chegou em 2008, e memória confiável e com cache, que veio com a arquitetura Fermi, em 2010.
FP32 tem 24 bits de mantissa, FP64 tem 53. E as unidades de FP64 são separadas das de FP32: quantas cada chip tem depende de para quem ele é vendido. O nosso laptop tem 2 por SM. A A30 do IAG tem 32. Isso vai voltar no fim, na comparação entre as duas.
▶ Clique: o conceito. Quadro: precisão; hierarquia de memória.

## 2012–2017
• AlexNet (2012) em duas GTX 580
• redes neurais = produtos de matrizes
• tensor cores (2017): o SM vira unidades especializadas
Em 2012, a AlexNet, a rede neural que reabriu a área, foi treinada em duas placas de jogos, duas GTX 580. Redes neurais são, no fundo, produtos de matrizes: vazão pura.
Em 2017 vieram os tensor cores, unidades especializadas em produtos de matrizes. A partir daí, o SM é um conjunto de unidades diferentes: FP32, inteiros, FP64 e tensor.
O nosso kernel usa sobretudo inteiros e FP32: o gerador aleatório, que vamos ver, é todo de inteiros.
▶ Clique: o conceito. Quadro: FP32, INT32, FP64, tensor.

## Hoje
• Frontier (2022): exaescala, máquina de GPUs
• o limite é energia por operação
• no laptop: FP32 em 36 SMs roda a 2730 MHz
Hoje, a Frontier, de 2022, o primeiro supercomputador a passar de um exaflop, é uma máquina de GPUs. E o limite deixou de ser o número de transistores: é a energia por operação.
Isso aparece até no nosso laptop. Quando eu uso todos os SMs em FP32, o driver baixa o clock para caber na potência. Medimos: cai de uns 2890 para 2730 MHz.
E nos usos da ciência está a transferência radiativa Monte Carlo, que é exatamente o EP4.
▶ Clique: os usos na ciência.

## De extensão do C a ecossistema
• linguagens, bibliotecas, ferramentas, aplicações
• nós usamos: cuRAND, Nsight Systems
• OpenACC: outro grupo
CUDA não é só uma extensão do C. É um ecossistema: linguagens, de C++ a Python; bibliotecas, como cuBLAS e cuFFT; ferramentas de perfil; e aplicações. É isso que faz uma plataforma durar vinte anos.
Em negrito está o que eu usei: o cuRAND, o gerador de números aleatórios dos dois EPs, e o Nsight Systems, com que medi o EP3.
E OpenACC, o jeito de chegar à GPU por diretivas, é o tema de outro grupo da turma.
▶ Clique: o conceito.
