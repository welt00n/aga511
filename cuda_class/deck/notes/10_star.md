## divisor
• Parte 9, ~25 min, a maior
• premissa, desenvolvimento, validação, resultados
• a resposta ao "como?" do gancho
Parte 9: o segundo projeto, o EP4, a estrela manchada. É a parte maior: premissa, desenvolvimento, validação e resultados. E é aqui que respondo o "como?" do começo.

## EP4: a estrela manchada
• estrela com escurecimento de limbo e manchas
• observadores em várias inclinações; quadro = fase da rotação
• algoritmo do EP2 (CPU) → GPU
A premissa. Uma estrela com escurecimento de limbo, mais escura na borda, e com manchas. Vários observadores a olham de inclinações diferentes, e cada quadro do vídeo é uma fase da rotação.
O algoritmo é o do EP2, que rodava sequencial na CPU. O EP4 leva esse algoritmo para a GPU.
Reparem: os observadores do sul não veem manchas. As três estão no hemisfério norte.

## A vida de um fóton
• 1 superfície (21%) · 2 direção, limbo (35%) · 3 mancha (9%)
• 4 observador e quadro (22%) · 5 pixel (1%) · registrar (13%)
• a estrela não gira: o azimute de escape é a fase
A vida de um fóton, em cinco passos, com a parte do tempo de cada um, medida.
Um: sorteio um ponto na superfície, 21% do tempo. Dois: sorteio a direção de saída, com o escurecimento de limbo, 35%, o passo mais caro, porque tem senos e cossenos. Três: se o ponto cai numa mancha, na sombra o fóton é bloqueado, na penumbra escapa com uma probabilidade; 9%. Quatro: a direção escolhe o observador e o quadro, 22%. Cinco: o pixel, 1%. E registrar na imagem, 13%.
E um truque bonito: a estrela não gira no código. O azimute para onde o fóton escapa é a fase da rotação.

## Do serial ao paralelo
• o esqueleto do Buffon, outro corpo
• o mesmo fluxo por pedaço; a física __host__ __device__
• o resultado: uma imagem de 5 GB
O desenvolvimento. É o esqueleto do Buffon, com outro corpo. O mesmo fluxo por pedaço, a mesma fila. A física é compilada para os dois processadores, então é uma função só, testada uma vez.
A grande diferença está na última linha: o resultado não é mais um número. É uma imagem de 5 gigabytes, onde cada fóton soma 1 num pixel.
▶ Cliques: o fluxo, o laço, o atômico na imagem.

## Onde mora cada coisa
• estrela: constante · fóton: registradores
• imagem: global, 1 atômico por fóton
• fila: 1 atômico por pedaço
A parte 7, aplicada. A estrela fica em memória constante, lida por todos. Cada fóton vive nos registradores do seu thread. A imagem fica na global, e cada fóton faz um atômico nela. E a fila de pedaços faz um atômico por pedaço.

## Como validamos os geradores
• 6 níveis, de baixo para cima
• 1–3: o gerador · 4: a divisão · 5: a física · 6: estatística
• tudo em main.py test (20 s) e no measure
A validação, em seis níveis, de baixo para cima. Cada nível só faz sentido se o de baixo passou.
O 1, o 2 e o 3 testam o gerador: a nossa Philox contra as respostas publicadas; o cuRAND da GPU contra a nossa; e as uniformes, bit a bit. O 4 testa a ideia do fluxo por pedaço: qualquer divisão do trabalho dá a mesma imagem. O 5 testa a física inteira: a imagem da GPU igual à da CPU. E o 6 testa com fótons diferentes, como um experimento.
Tudo isso roda no main.py test, em uns 20 segundos, e de novo, com N grande, no measure.
▶ Um clique por nível.

## A saída real
• não é afirmação: é a saída do programa
• 3 vetores de teste do Random123
• pedaço 0: mesmos 8 números na CPU e na GPU
• pedaço 2³² + 5: o vai-um
Isto não é uma afirmação, é a saída do programa. As três primeiras linhas são os vetores de teste publicados do Random123: passam.
As duas do meio: os oito primeiros números do pedaço 0, na CPU e na GPU. Iguais.
E o pedaço 4 294 967 301, que é 2 a 32 mais 5, testa o vai-um entre as palavras do contador.

## O nível 6
• mesma semente, pedaços diferentes
• χ²/gl = 0,990 em 51 899 blocos (−1,6σ)
• sementes diferentes = estrelas diferentes (primeira versão falhou)
O nível 6: fótons diferentes, da mesma estrela. Mesma semente, pedaços diferentes: a CPU fica com uma metade, a GPU com a outra. Comparando as imagens em blocos de 8 por 8 pixels, o qui-quadrado por grau de liberdade é 0,990, com quase 52 mil graus de liberdade.
Por que a mesma semente? Porque a semente também posiciona as manchas. Na primeira versão usei sementes diferentes e o teste falhou: outra semente é outra estrela.

## A validação da estrela
• FP64: GPU = CPU, bit a bit (10⁸ fótons)
• FP32: 3·10⁻⁴ dos fótons mudam de pixel
• pergunta: por que dois processadores dão os mesmos bits?
O resumo. Em FP64, a imagem da GPU é idêntica à da CPU, bit a bit, com 100 milhões de fótons. Em FP32, 3 em cada 10 mil fótons mudam de pixel.
Pergunta: por que dois processadores diferentes podem dar exatamente os mesmos bits?
▶ Clique: a pergunta. Esperar.
▶ Clique: a resposta. Os mesmos números aleatórios (parte 5); contagens inteiras (parte 7); e em FP64 as diferenças do último bit (parte 4) quase nunca põem um fóton do outro lado da borda de um pixel: nenhum em 10⁸. Em FP32, alguns.

## Dois eixos de custo
• resolução (quadros, bins) → fótons, borrão (~11 px; polo 105 px)
• tamanho → memória, passes (cada passe traça tudo de novo)
• nitidez de pixel: outro algoritmo (TCC)
Dois eixos de custo, que é bom separar. A resolução, quantos quadros e quantas faixas de inclinação, custa fótons e decide o borrão: a 1024 com 300 quadros, uns 11 pixels de borrão no equador, até 105 no polo.
O tamanho, pixels vezes quadros vezes observadores vezes 4 bytes, custa memória. Sete observadores a 1024 já pedem dois passes no laptop, e cada passe traça todos os fótons de novo.
Imagens nítidas ao nível do pixel pediriam outro algoritmo. Isso fica para o meu TCC.

## EP4: para onde vai o tempo
• fótons: 216,9 s → 6,0 s (36,3×)
• o resto não encolhe (salvar 5 GB)
• pergunta: e a execução inteira?
• 19,3×
Os resultados de Amdahl. Em FP64, com 4 bilhões de fótons e a imagem de 5 GB: os fótons levam 216,9 segundos num SM e 6,0 em 36. 36,3 vezes.
Mas preparar, copiar de volta e salvar 5 GB no disco não encolhem com os SMs.
Então, quanto deve dar a execução inteira?
▶ Clique: o resto. Clique: a pergunta; esperar. Clique: 19,3×. Clique: a frase.

## As fases da execução
• 1 SM: fótons 97% · 36 SMs: fótons 52%
• salvar 5 GB: ~4,3 s em qualquer caso
• a parte serial é a própria imagem
A mesma coisa, em figura. Num SM, os fótons são 97% do tempo. Em 36, só 52%. Salvar 5 GB leva uns 4,3 segundos, com um SM ou com 36.
Calcular é barato; mover dados não é. Numa imagem de alta resolução, a parte serial é a própria imagem.

## O teto de Amdahl
• azul: fótons, quase ideal · laranja: execução inteira → teto 38×
• verde: em ciclos, sem o clock
• FP64 ~2880 MHz; FP32 em 36 SMs: 2730 (potência)
Azul: os fótons, quase no ideal. Laranja: a execução inteira, curvando para um teto de 38 vezes, quantos SMs houver. Ninguém passa disso com esta imagem.
Verde: os fótons medidos em ciclos, o que tira o clock da conta. À direita, o clock durante os fótons: em FP64 fica em uns 2880 MHz. Em FP32 com todos os SMs ele cai para 2730, pelo limite de potência: 35,3 vezes em segundos, 36,3 em ciclos.

## Quanto maior N
• T = N/velocidade + parte serial (ambos medidos)
• medidos sobre a curva (FP32 10¹¹: 21,4× × 22,2×)
• o p(N) do Buffon de novo
E quanto maior N, mais perto do ideal. O modelo é simples: o tempo é N dividido pela velocidade, mais a parte serial, os dois medidos. Os pontos medidos caem na curva: em FP32 com 10 a 11 fótons, 21,4 vezes medido, 22,2 previsto.
É o mesmo p que cresce com N do Buffon: a parte serial é fixa, a paralela cresce.

## A mesma lei, outra parte serial
• EP3: 0,15 s (contexto) → 34×
• EP4: 5,6 s (salvar 5 GB) → 19×
• pergunta: 2048², melhor ou pior?
A mesma lei de Amdahl, duas partes seriais diferentes. No EP3, 0,15 segundo de contexto: 34 vezes. No EP4, 5,6 segundos, sobretudo salvar a imagem: 19 vezes.
Pergunta: com uma imagem de 2048, a execução inteira escala melhor ou pior?
▶ Clique: a pergunta. Resposta: pior para o mesmo N (imagem 4× maior para salvar); melhor só se N crescer junto (Gustafson).

## A mesma estrela em 2048²
• 1,56·10¹³ fótons; 1 h 43 min no laptop
• 4 passes (20 GB > 8 GB)
• SNR medida = prevista: 14,0 / 11,7
• A30: 24 GB, 1 passe · [A30]
A mesma estrela em 2048 por 2048. 15,6 trilhões de fótons: 93 minutos de fótons, uma hora e quarenta e três no total, no laptop.
Foram 4 passes: 20 GB de imagem não cabem em 8, e cada passe traça os fótons de novo, guardando só alguns observadores.
A relação sinal-ruído medida bate com a prevista: 14,0 no centro, 11,7 na borda.
A A30 tem 24 GB: um passe só. [A30: o tempo dela.]

## Mais fótons, imagem mais limpa
• SNR = √(fótons por pixel)
• 10⁹ → 10¹³: SNR na borda 0,19 → 18,7, todos a < 1% do previsto
• manchas a partir de 10¹⁰
• degraus = somas parciais (fluxo por pedaço)
O que os fótons compram. A relação sinal-ruído é a raiz do número de fótons por pixel. De um bilhão a dez trilhões de fótons, a SNR na borda vai de 0,19 a 18,7, cada degrau a menos de 1% do previsto. As manchas aparecem a partir de 10 a 10.
E cada degrau é exatamente o começo do seguinte: são somas parciais. Isso só é possível porque o fluxo aleatório é por pedaço.
▶ Deixar o vídeo rodar uma volta.

## A A30: Amdahl em 56 SMs
• [A30: figura + números]
• fótons: FP64 rápido; execução inteira: o disco do IAG
[A30: trocar o quadro tracejado pela figura da A30 e preencher.]
O que esperar na comparação: nos fótons, a A30 tem FP64 rápido; na execução inteira, o teto depende do disco do IAG, que é onde a imagem é salva.

## O laptop e a A30
• a resposta da parte 2: FP64? FP32?
• por SM por GHz: o que um SM faz por ciclo
• [A30: main.py compare]
Agora a resposta à pergunta da parte 2: qual das duas ganha em FP64, e em FP32.
A linha "por SM por GHz" tira da conta o número de SMs e o clock, e deixa o que um SM faz por ciclo.
[A30: preencher com main.py compare.]

## Uma comparação justa
• "100× mais rápida que a CPU": GPU inteira × 1 núcleo
• erro clássico (Lee et al., ISCA 2010)
• aqui: 1 SM × todos, mesmo código, mesmo N
• sempre dizer contra o quê
Uma última coisa sobre medir. É comum ler "a GPU é cem vezes mais rápida que a CPU". Quase sempre isso compara uma GPU inteira com um núcleo de um código sequencial. É um erro clássico, descrito num artigo de 2010 chamado "desmistificando o mito dos 100×".
Aqui eu comparo a GPU com ela mesma: um SM contra todos, o mesmo código, o mesmo N. A CPU serve para outra coisa: conferir que a resposta está certa.
E sempre dizer contra o quê: fótons ou execução inteira, segundos ou ciclos.
▶ Um clique por item.
