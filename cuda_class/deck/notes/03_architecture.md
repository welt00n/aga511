## divisor
• Parte 2, ~6 min
• por que milhares de núcleos simples; o que há num SM
Parte 2: arquitetura. Duas perguntas: por que uma GPU tem milhares de núcleos simples, e o que há dentro de um SM.

## CPU e GPU
• CPU: controle + cache, uma tarefa rápida (latência)
• GPU: unidades de cálculo, muitas tarefas (vazão)
• laptop: 36 × 128 = 4 608 faixas FP32
• Monte Carlo: milhões de tarefas iguais e independentes
À esquerda, uma CPU: poucos núcleos grandes, e boa parte do chip gasta em controle e em cache. Tudo isso serve para que uma tarefa termine o mais rápido possível: é feita para a latência.
À direita, a GPU: o chip é quase todo unidades de cálculo, agrupadas em SMs. Cada tarefa sozinha é lenta, mas milhares andam juntas: é feita para a vazão. O próprio guia da NVIDIA diz isso: a GPU troca o desempenho de um thread pela vazão total.
No meu laptop são 36 SMs com 128 faixas FP32 cada: 4 608 faixas. E Monte Carlo é o caso perfeito para isso: milhões de tarefas iguais e independentes.

## O que há dentro de um SM
• 4 partições: escalonador, 32 faixas, tensor, 16 384 registradores
• o SM inteiro: 2 unidades FP64 (laptop) × 32 (A30)
• registradores iguais; warps: 48 × 64
Este é um SM do meu laptop, arquitetura Blackwell. Ele tem quatro partições. Cada uma tem o seu escalonador de warps, 32 faixas que fazem FP32 ou inteiros, um tensor core, as unidades de memória e de funções especiais, e 16 384 registradores.
Agora reparem embaixo: para o SM inteiro, só duas unidades de FP64. É uma placa de jogos. A A30, que é uma placa de ciência, tem 32.
Na tabela da direita, as duas GPUs: os mesmos 65 536 registradores por SM, mas a A30 comporta 64 warps, e o laptop 48.
▶ Apontar o 2 e o 32, em laranja: isso volta no fim.

## As nossas duas GPUs
• laptop: 36 SMs, cc 12.0, FP64 : FP32 = 1 : 64
• A30 (pituna, IAG): 56 SMs, cc 8.0, 1 : 2
• pergunta: qual ganha em FP64? E em FP32?
As duas GPUs desta aula. A RTX 5070 do meu laptop: Blackwell, 36 SMs, compute capability 12.0. E a A30 do IAG, a máquina se chama pituna: Ampere, 56 SMs, compute capability 8.0, 24 GB de memória.
A razão entre FP64 e FP32 é de 1 para 64 no laptop e de 1 para 2 na A30.
Então pergunto: qual das duas ganha em FP64? E em FP32?
▶ Clique: a pergunta. Deixar a turma apostar; anotar as apostas no quadro.
▶ Dizer: a resposta, medida, vem na parte 9.
