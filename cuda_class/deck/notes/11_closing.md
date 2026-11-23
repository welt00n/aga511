## divisor
• Parte 10, ~5 min
Parte 10: o fechamento. O fio de volta, e como reproduzir cada número desta aula.

## O fio, de volta
• a tabela do começo, atravessada
• ler 2 ou 3 linhas
Lembram da tabela do começo? Agora cada linha foi atravessada. Os pixels independentes de 1999 viraram um fóton por iteração. O SM de 2006 virou a unidade de toda aceleração. O FP64 de 2008 virou 16 vezes mais lento aqui, e outra coisa na A30. E os números aleatórios em paralelo viraram a mesma imagem na CPU e na GPU.
▶ Ler duas ou três linhas, não todas.

## Três lições
• 1 muitos pedaços independentes
• 2 primeiro a correção: os mesmos bits
• 3 Amdahl decide: ache a parte serial, diga contra o quê
Se ficarem três coisas desta aula:
Um: a GPU vence quando o trabalho são muitos pedaços independentes.
Dois: primeiro a correção. Antes de medir velocidade, os mesmos bits na CPU e na GPU.
Três: Amdahl decide. Achem a parte serial, seja o contexto ou a imagem, meçam a execução inteira, e digam sempre contra o quê cada aceleração foi medida.
▶ Um clique por lição, devagar.

## Cada número desta aula sai de um comando
• engine (C++/CUDA) = o projeto
• orquestrador (Python) = como é apresentado e analisado
• test · make · measure · compare
O engine em C++ e CUDA é o projeto. O orquestrador em Python é como ele é apresentado e analisado. Todos os gráficos desta aula saíram destes quatro comandos, no laptop e na A30.
E antes de rodar, cada um confere o ambiente, prevê tempo, memória e disco, e só aceita um N válido.

## Para levar
• o guia, com o nosso código: 4 seções
• cada uma: seção do guia, código, número, comando
O material para levar: o guia, com o nosso código. As quatro partes do guia que usamos hoje, cada uma com a seção para citar, o nosso código, o nosso número medido, e o comando que o reproduz.
▶ Mostrar onde está: o repositório.

## Obrigado
• perguntas
Obrigado. Perguntas?
▶ Ter à mão: as duas GPUs (parte 2), o teto de Amdahl, a pirâmide da validação, threads persistentes.
