## divisor
• Parte 4, ~8 min
• guia 5.6, medido no Buffon
Parte 4: o segundo capítulo do guia, ponto flutuante. Quanto custa e quanto vale cada precisão, medido no Buffon.

## FP16, FP32 e FP64
• IEEE 754: sinal, expoente, mantissa
• ~3, ~7, ~16 dígitos
• unidades separadas: 128 FP32 × 2 FP64 por SM (laptop)
• a pergunta: o erro é menor que o ruído?
Os três formatos do padrão IEEE 754. FP16 guarda uns 3 dígitos, FP32 uns 7, e FP64 uns 16. O ε é o erro relativo de uma operação. O guia diz que todas as GPUs seguem o IEEE 754.
E as unidades são separadas no SM: no laptop, 128 faixas de FP32 e só 2 de FP64.
A pergunta certa, num Monte Carlo, não é "qual é mais exato". É: o erro de arredondamento é menor que o ruído estatístico?
▶ Clique: as unidades. Clique: a pergunta certa.

## O mesmo Buffon em FP64, FP32 e FP16
• FP32 16× mais rápido: 6 casas em 51 s (FP64: 14 min)
• FP32 nunca chega à 7ª casa: viés ~10⁻⁷
• FP16: 3 casas
• não é erro acumulado; é viés
O EP3 tem três versões da mesma conta. FP32 é 16 vezes mais rápido que FP64: seis casas de π saem em 51 segundos, em vez de 14 minutos.
Mas o FP32 nunca chega à sétima casa. A comparação "x menor que L sobre 2 vezes seno de θ", feita em 24 bits, tem um viés de uns 10 a menos 7. Quando o erro estatístico cai abaixo disso, mais agulhas não ajudam. E o FP16 para em 3 casas.
E não é erro acumulado: cada agulha é uma conta curta e independente. É um viés da comparação.
Pergunta: e a estrela, precisa de FP64?
▶ Cliques: velocidade, viés, não acumulado. Clique: a pergunta.
▶ Resposta, adiantando a parte 9: não; FP32 muda o pixel de 7·10⁻⁴ dos fótons, abaixo do ruído.

## Por que a CPU e a GPU discordam no último bit
• FMA: a·b + c com um arredondamento só (padrão na GPU)
• funções: sin/cos 2 ULP, atan2f 3, cbrt 1; implementações diferentes
• sem FMA: FP64 a mesma imagem; FP32 outra
Por que a mesma conta dá resultados diferentes no último bit na CPU e na GPU? Duas razões, as duas no guia.
A primeira é o FMA: a vezes b mais c com um arredondamento só, em vez de dois. O compilador da GPU faz isso por padrão.
A segunda: as funções matemáticas são implementações diferentes. O guia dá até 2 ULP de erro para seno e cosseno, 3 para o atan2 em FP32. E diz, literalmente, que diferenças não significam que a GPU esteja errada.
Medimos: compilando sem FMA, em FP64 a imagem da estrela é a mesma; em FP32, é outra.
▶ Cliques: a medida, a consequência.

## Precisão também é escolher a conta
• 1 − cos² perto do polo: cancelamento
• (1 − cos)(1 + cos): mesma conta, até ~400× melhor
• manchas: corda ao quadrado
• Monte Carlo perdoa: contagem inteira; cuidado com viés
E precisão não é só escolher o tipo. É também escolher a conta. Este é o nosso código: para tirar o seno a partir do cosseno, perto do polo o cosseno vai a 1, e 1 menos cosseno ao quadrado subtrai dois números quase iguais, perdendo justamente os bits que importam.
Escrevendo um menos cosseno vezes um mais cosseno, é a mesma conta, com o mesmo custo, e a diferença pequena fica exata: até umas 400 vezes melhor no pior caso. Nas manchas fiz o mesmo: comparo a corda ao quadrado, em vez de dois cossenos perto de 1.
E a boa notícia: Monte Carlo perdoa muito. Cada fóton é uma conta curta, e a imagem é uma contagem inteira. O que importa é não introduzir viés.
▶ Clique: destaca a linha. Cliques: cancelamento, conta certa, manchas, conclusão.
