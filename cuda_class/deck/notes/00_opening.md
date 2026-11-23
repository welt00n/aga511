## CUDA: programando a GPU
• CUDA + os dois EPs dentro da aula
• cada conceito chega quando um projeto precisa dele
Boa tarde. Hoje a aula é de CUDA, a plataforma da NVIDIA para programar a GPU. E os dois projetos que fiz com ela, o EP3, a agulha de Buffon, e o EP4, a estrela manchada, não vêm no fim: eles estão dentro da aula.
A ideia é simples: cada conceito de CUDA aparece no momento em que um dos projetos precisa dele. Então, quando eu falar de blocos, de memória ou de ponto flutuante, vocês vão ver isso funcionando num código de verdade, com números medidos.
▶ Ligar o relógio do tablet com o primeiro ▶.

## Roteiro
• 10 partes, 3 blocos
• fundamentos → guia da NVIDIA com o nosso código → projetos
• 2 capítulos do guia antes do Buffon, 2 antes da estrela
A aula tem dez partes, em três blocos. Primeiro os fundamentos: um pouco de história e a arquitetura da GPU.
Depois, o guia oficial da NVIDIA, o CUDA Programming Guide. Vou passar por quatro capítulos dele, mas sempre usando o nosso código como exemplo. Dois capítulos vêm antes do Buffon: o modelo de execução, com SPMD e SIMT, e o ponto flutuante. Os outros dois vêm antes da estrela: a memória e a ocupação.
E o fio de tudo está escrito embaixo: um conceito nasce de um problema, é explicado, e depois é medido num projeto.
▶ Não ler a lista. Apontar os três blocos.

## O fio da aula
• cada linha: uma ponte
• exemplo: o SM nasce em 2006 → por dentro → unidade de toda aceleração
• voltamos aqui no fim
Esta tabela é o mapa das pontes que vamos atravessar. Cada linha é um conceito: onde ele nasceu na história, onde vou explicá-lo, e onde ele aparece medido num dos projetos.
Um exemplo: o SM, o multiprocessador. Ele nasce em 2006, com a G80. Na parte 2 vamos ver o que há dentro dele. E no fim ele vira a unidade de toda aceleração que eu meço: um SM contra trinta e seis SMs.
Não vou ler a tabela agora. Vamos voltar a ela no fechamento, e aí cada linha vai fazer sentido.
▶ Escolher uma linha (o SM) e contar a ponte inteira, como exemplo.
