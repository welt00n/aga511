## divisor
• Parte 6, ~14 min
• o EP3 como entregue: premissa, desenvolvimento, resultados
Parte 6: o primeiro projeto, o EP3, a agulha de Buffon, do jeito que foi entregue. Premissa, desenvolvimento e resultados.

## Ponte para o EP3
• pergunta: contar agulhas que cruzam = que operação?
• resposta: uma redução
Antes de começar, uma pergunta. O Buffon conta quantas agulhas cruzam uma linha. Que operação é essa, do que vimos até agora?
É uma redução. Cada thread soma as suas agulhas em registrador, e um atômico junta tudo no fim.
▶ Ler a frase e esperar respostas. Clique: a resposta.

## EP3: a agulha de Buffon
• agulha L, linhas a distância D
• cruza se x ≤ (L/2) sin θ
• P = 2L/(πD) → π ≈ 2Ln / (D · cruzamentos)
A premissa. Jogamos agulhas de comprimento L sobre um chão com linhas paralelas a distância D. Cada agulha tem o centro a uma distância x da linha mais próxima e faz um ângulo θ com as linhas. Ela cruza se x for menor ou igual a L sobre 2 vezes o seno de θ.
A probabilidade de cruzar é 2L sobre π D. Então, contando os cruzamentos, estimamos π. No desenho, em azul, as que cruzam.
Dois números aleatórios por agulha: x e θ.

## EP3: o código serial
• um laço: sorteia, testa, conta
• cada agulha independente = paralelismo de dados
• contar = redução
O código serial é um laço: sorteia x, sorteia θ, testa, conta.
E as duas pontes com o que já vimos: cada agulha é independente das outras, é paralelismo de dados, como os pixels de 1999. E contar é uma redução, como na parte 3.

## EP3: paralelizar com threads persistentes
• tantos threads quanto cabem, lançados uma vez
• fila de pedaços: atomicAdd no contador
• soma em registrador, 1 atômico por thread no fim
• quem termina antes pega mais
O desenvolvimento. Lanço tantos threads quanto cabem nos SMs, uma vez só. Cada thread pega o próximo pedaço de agulhas de uma fila, com um atomicAdd num contador global, e cada pedaço tem o seu fluxo Philox.
O thread soma os cruzamentos dele num registrador, e só no fim faz um atômico no total.
Quem termina antes pega mais pedaços: o trabalho se equilibra sozinho.

## EP3: o mesmo laço na GPU
• fluxo Philox por pedaço (parte 5)
• parcial em registrador + 1 atômico (parte 3)
• a fila (parte 3)
O código real. Três ideias num kernel só: o fluxo Philox por pedaço, da parte 5; a redução, com o resultado parcial em registrador e um atômico por thread, da parte 3; e a fila de pedaços, também da parte 3.
▶ Cliques: o fluxo, o laço, o atômico.

## EP3: o erro cai
• erro ∝ N^(−0,499 ± 0,009): valida o gerador
• tempo: 0,15 s fixos (contexto) + proporcional a N
Os primeiros resultados. À esquerda, o erro da estimativa de π cai com N elevado a menos 0,499, mais ou menos 0,009. A teoria diz menos meio. Isso também valida o gerador: se os números tivessem correlação, o erro não cairia assim.
À direita, o tempo: 0,15 segundo fixos, que é criar o contexto CUDA, mais uma parte proporcional a N. Para N pequeno, o tempo é só o contexto.

## EP3: os resultados
• π = 3,1415942 ± 0,0000024 (0,6σ), 9,7·10¹¹ agulhas
• k = 2,38 ± 0,10 (teoria 2,374)
• FP64 6,6·10⁹/s; FP32 1,07·10¹¹/s
• 14 casas: 2,7·10¹¹ anos
Os resultados em uma tabela. Juntando todas as agulhas, quase um trilhão: π igual a 3,1415942, mais ou menos 0,0000024. O valor verdadeiro está a 0,6 sigma.
O erro segue a teoria: o coeficiente medido é 2,38, a teoria diz 2,374.
A velocidade: 6,6 bilhões de agulhas por segundo em FP64, 107 bilhões em FP32.
E a pergunta do EP1, quanto tempo para 14 casas de π: 5,6 vezes 10 a 28 agulhas, 270 bilhões de anos nesta GPU. Esse número nem cabe num inteiro de 64 bits.
▶ Clique: a pergunta. Passa para Amdahl.

## A lei de Amdahl, com SMs
• S(n) = 1 / (s + p/n); teto 1/s
• n = SMs (a máscara)
• a pergunta: qual é a parte serial de um programa de GPU?
Amdahl vocês conhecem: S de n é 1 sobre s mais p sobre n, e o teto é 1 sobre s, quantos processadores houver. No desenho, com s igual a 0,1, o teto é 10.
A novidade aqui é que n são SMs, graças à máscara da parte 3. E a pergunta que importa: o que é a parte serial de um programa de GPU?

## EP3: Amdahl nos SMs
• pergunta antes: 36 SMs, quanto?
• 34,3× em 36 SMs, p = 0,9986 (N = 3,2·10¹⁰)
• serial: 0,15 s = o contexto CUDA
• p cresce com N: 0,07 → 0,78 → 0,997
Antes de mostrar: em 36 SMs, quanto mais rápido vocês acham que fica?
Com 32 bilhões de agulhas em FP64: 34,3 vezes em 36 SMs. O p ajustado é 0,9986. A parte serial é aqueles 0,15 segundo: criar o contexto CUDA.
E um ponto importante: o p não é uma propriedade do programa. Ele cresce com N: 0,07 com um milhão de agulhas, 0,78 com cem milhões, 0,997 com dez bilhões. A parte serial é fixa, e a paralela cresce.
Guardem o 0,15 segundo: a estrela vai ter outra parte serial.
▶ Clique: a pergunta; anotar apostas. Clique: o resultado.
