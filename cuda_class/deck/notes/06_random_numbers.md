## divisor
• Parte 5, ~8 min
• cada pedaço do trabalho precisa dos seus números
Parte 5: números aleatórios em paralelo. Monte Carlo vive deles, e em paralelo cada pedaço do trabalho precisa dos seus, sem repetir os dos outros.

## Quantos números
• Buffon 2 por agulha; estrela 5 ou 6 por fóton
• gancho: ~2·10¹³ números
• qualidade, independência, reprodutibilidade
Quantos números? O Buffon usa dois por agulha. A estrela usa cinco ou seis por fóton: dois para o ponto na superfície, três para a direção, e às vezes um para a mancha. A imagem do gancho consumiu uns 2 vezes 10 a 13 números.
Três exigências: qualidade, que pareçam independentes; independência entre threads, nenhum trecho repetido; e reprodutibilidade: a mesma semente tem que dar a mesma imagem, em qualquer processador.
▶ Clique: as três exigências.

## O gerador clássico tem estado
• LCG: o próximo sai do anterior (drand48 do EP3 na CPU)
• para chegar a x₁₀₀₀, passar por todos: serial
• em paralelo: quem usa qual trecho? correlação; depende da divisão
O gerador clássico tem estado. Um LCG, como o drand48 que o EP3 usava na CPU: cada número sai do anterior, multiplicando e somando, módulo 2 a 48.
Para chegar ao milésimo número, é preciso passar por todos os anteriores. A sequência é serial.
Em paralelo isso vira um problema: quem usa qual trecho? Se cada thread usar uma semente vizinha, as sequências podem sair correlacionadas. E o resultado passa a depender de quantos threads havia e de quem pegou o quê.
▶ Clique: os três problemas.

## Um contador em vez de um estado: Philox
• número n = f(chave, n): qualquer n, direto
• Philox4x32-10 (Salmon et al., 2011): 4 números de 32 bits por contador
• bijeção; passa no BigCrush
• o mesmo na CPU (nosso) e na GPU (cuRAND)
A solução é não ter estado. Num gerador baseado em contador, o número n é uma função de n e de uma chave. Dá para pedir qualquer n, direto, em qualquer ordem, em qualquer processador.
O que eu uso é o Philox, de Salmon e coautores, de 2011. Ele embaralha um contador de 128 bits com uma chave de 64, e devolve quatro números de 32 bits. É uma bijeção, contadores diferentes nunca dão a mesma saída, e passa na bateria BigCrush, que é o teste mais exigente de geradores. Isso é resultado publicado, eu não rodei.
É o mesmo algoritmo nos dois lados: na CPU, uma implementação nossa; na GPU, o cuRAND.
▶ Cliques: Philox, bijeção, CPU e GPU.

## Uma rodada de Philox
• multiplicação 32 × 32 → 64, alta e baixa
• XOR com a chave, palavras trocam de lugar
• 10 rodadas, tudo inteiro: mesmos bits em qualquer lugar
Uma rodada do Philox, para quem gosta de ver por dentro. As palavras x e z são multiplicadas por constantes, de 32 por 32 bits dando 64, divididos em parte alta e baixa. As partes altas fazem XOR com a chave e com as outras palavras, e tudo troca de lugar.
São dez rodadas, e entre elas a chave soma constantes fixas. É tudo aritmética inteira exata. Por isso a CPU e a GPU dão exatamente os mesmos bits.

## Um fluxo por pedaço de 1024 fótons
• chave = semente; contador = (pedaço, posição)
• ~6 000 números por pedaço, cabem 2⁶⁶
• curand_init(seed, chunk, 0) = philox_init(seed, chunk, 0)
• o pedaço escolhe os números, não o thread
Como uso isso. A chave é a semente. O contador tem duas metades: o número do pedaço, 64 bits, e a posição dentro do pedaço, outros 64. Um pedaço de 1024 fótons usa uns 6 mil números, e cabem 2 a 66: os fluxos de dois pedaços nunca se encontram.
Na GPU, curand_init com a semente e o pedaço; na CPU, a nossa philox_init com os mesmos argumentos. É a mesma coisa.
E esta é a decisão central do EP4: quem escolhe os números é o pedaço, não o thread. Então quem traça um pedaço, que SM, que thread, CPU ou GPU, não muda a imagem.
▶ Clique: a conclusão. Dizer: guardem isso; é o que torna a estrela bit a bit.

## Os números viram uniformes
• u = x · 2⁻³² + meio passo: nunca 0, pode ser 1
• a conversão do cuRAND; conferida bit a bit
• limite: 2³² valores (passo 2,3·10⁻¹⁰), muito abaixo de um pixel
O inteiro de 32 bits vira uma uniforme assim: x vezes 2 a menos 32, mais meio passo. Nunca dá zero, pode dar um. É a conversão do cuRAND, e a nossa CPU repete a mesma conta.
Um limite conhecido: mesmo em FP64, há só 2 a 32 valores possíveis, um passo de 2,3 vezes 10 a menos 10. Para a estrela, isso é milhões de vezes menor que um pixel.
▶ Cliques: a conversão, o limite.
