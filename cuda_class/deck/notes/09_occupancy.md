## divisor
• Parte 8, ~10 min
• quantos warps cabem; o guia recomenda; nós medimos
Parte 8: o quarto capítulo do guia, a ocupação. Quantos warps cabem num SM, o que o guia recomenda, e o que medimos.

## Esconder a espera trocando de warp
• 1 warp: a unidade parada 75% do tempo
• 4 warps: sempre alguém pronto
• trocar de warp custa zero
• só ajuda se a espera for latência
Um warp emite uma instrução e depois espera o resultado: da memória, ou de uma unidade lenta. Com um warp só, a unidade fica parada boa parte do tempo. Com quatro warps, enquanto um espera, outro está pronto, e a unidade fica sempre ocupada.
É assim que a GPU esconde latência, não com caches grandes como a CPU. E trocar de warp não custa nada, porque o estado de todos fica nos registradores.
Mas atenção: isso só ajuda se a espera for latência. Se a unidade já está cheia, mais warps só fazem fila.
▶ Clique: os 4 warps. Clique: a ressalva.

## Os registradores decidem quantos warps cabem
• 65 536 registradores por SM, em passos de 8 por thread
• FP64: 80 → cabem 25, lançamos 24 → 0,50
• FP32: 67 → 72 → 28 → 0,58
• A30 FP64: 96 → 21 de 64 → 0,33
Quantos warps cabem num SM? Quem decide são os registradores. O SM tem 65 536, e eles são dados por warp, em passos de 8 por thread.
No nosso FP64, 80 registradores por thread: 2 560 por warp. Cabem 25 warps, mas lançamos 24, porque o bloco só entra inteiro. Ocupação: 24 de 48, 0,50.
Em FP32, 67 arredonda para 72: cabem 28, ocupação 0,58.
E na A30, o mesmo código compilado para ela pede 96 registradores: 21 warps de 64, no máximo 0,33.

## O nosso lançamento, como o orquestrador o imprime
• todo produto de GPU imprime esta linha
• o bloco vem da API de ocupação
• nenhum if por GPU · [A30: a linha dela]
Esta é a linha que o orquestrador imprime antes de qualquer execução na GPU: registradores, warps que cabem, threads por bloco, ocupação, total de threads.
O tamanho do bloco vem da API de ocupação do CUDA. Na A30, a mesma chamada escolhe o lançamento dela: não tem nenhum if por GPU no código.
[A30: ler aqui a linha que ela imprimiu.]

## O guia diz, nós medimos
• guia 2.3.7: ocupação "tão alta quanto possível"
• teste: -maxrregcount 80 → 32; ocupação 0,50 → 1,00
• medir em ciclos
• pergunta: fica mais rápido?
O guia, na seção 2.3.7, diz: é uma boa prática ter a ocupação tão alta quanto possível, porque isso esconde a latência e aumenta o desempenho. E diz que limitar os registradores pode trazer um ganho líquido.
Então testamos. Compilei o kernel limitando os registradores, de 80 até 32. A ocupação sobe de 0,50 até 1,00. E medi a velocidade em ciclos, para o clock não atrapalhar.
Pergunta: fica mais rápido?
▶ Clique: a pergunta. Deixar a turma apostar; a maioria diz que sim.

## Mais ocupação não é mais rápido
• não
• FP64: igual até 0,67; depois até 10% mais lento
• FP32: o mais rápido é sem limite
Não. No gráfico de cima, a ocupação sobe exatamente como a conta manda. No de baixo, a velocidade em ciclos não melhora.
Em FP64 fica igual até ocupação 0,67, e depois piora, até 10% mais lento. Em FP32, o mais rápido é sem limite nenhum.

## Por quê
• antes de derramar: o compilador recalcula (FP32 a 55: +4,7% instruções, +1,1% ciclos)
• depois: cada derramamento = escrita + leitura na local
• o SM não estava esperando latência
• sem sincronização envolvida
Por quê? A ocupação em si não custa nada. O que custa é tirar registradores do compilador, em dois passos.
Primeiro, antes de derramar qualquer coisa, o compilador passa a recalcular valores em vez de guardá-los. Em FP32 com 55 registradores, sem derramar nada, o kernel já tem 4,7% mais instruções, e leva 1,1% mais ciclos.
Depois vêm os derramamentos: cada valor que não cabe vira uma escrita e uma leitura na memória local, que mora na memória do device.
E mais warps só pagariam se o SM estivesse parado esperando latência. Aqui não está. E não há sincronização envolvida: o kernel não tem __syncthreads.
▶ Um clique por item.

## FP64 é 16× mais lento
• pico 1 : 64; medido 16,2×
• o kernel também é inteiro (Philox, índices, bins)
• 1 063 NOPs em 2 744 instruções: espera pelas 2 unidades FP64
• a fila é na unidade, não na memória
• pergunta: e na A30?
O laptop tem 2 unidades FP64 contra 128 faixas FP32: a razão de pico é 1 para 64. Mas medimos só 16 vezes. Por quê? Porque o kernel não é só FP64: o Philox, os índices e os bins são contas de inteiros, que não mudam com a precisão.
E olhando o código de máquina em FP64: 1 063 instruções NOP em 2 744. São esperas: o compilador faz o warp esperar as duas unidades FP64 terminarem. Isso veio de uma investigação nossa, não do orquestrador.
Por isso a ocupação não ajuda em FP64: a fila é na unidade de cálculo, não na memória.
E na A30, com 32 unidades FP64 por SM?
▶ Clique: a pergunta. Resposta esperada: perto de 1 a 2 vezes; a medida está na parte 9.

## E a A30?
• 96 / 76 registradores → ocupação ≤ 0,33 / 0,39
• FP64 rápido: a fila some, a latência pode importar
• a medida decide · [A30]
O mesmo código compilado para a A30 pede 96 registradores em FP64 e 76 em FP32. A ocupação fica no máximo em 0,33.
Mas o FP64 dela é rápido. A fila na unidade FP64 some, e a latência pode voltar a importar. Aí, mais ocupação talvez ajude. Quem decide é a medida.
Por que 96 e não 80? O compilador do CUDA 13 usa um front end mais novo só para Blackwell em diante. Também é investigação nossa.
[A30: ocupação medida, registradores mais rápidos na varredura, FP32 sobre FP64.]
