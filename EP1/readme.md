# Buffon's needle

Pretty cool embarrassingly parallel problem

## The important stuff

Imagine a infinite plan. Now, the x coordinate of this plan can be split in many segments of fixed length, lets call it d.


The experiment is quite simple, we get a needle of size l and throw it on this plane. What are the possible outputs of it? 


The needle may fall in any position and at any angle, but we are interested on a simple test: Does it cross one of the lines from the x axis segements? 


the important info is: how far from the closest line, lets call it x, and whats the angle, lets call teta, that the needle falls? depending on the angle and on how far the needle fell, we may check if it crossed the line or not.


the distance from the closest line is at most d/2  and the possible angles are 0 to pi/2 , all other values are equivalent because of the simetries of the problem, say we get a -pi/2 teta angle, that would bring give the same result as pi/2, the same happens with teta=pi or teta=pi/3, the simmetry allows us to focus on values normalized and make the possible set of tetas to be from 0 to pi/2. (little redundant this piece of text, i shall review all of this of course)


Now, how to check if the needle crosses? we calculate the lenght of the needle in the x axis using extension=(L/2)*sin(teta), if x, the distance from the line is smaller than (L/2)*sin(teta), the needle crosses a line.


Now, we can do the fun stuff and infer Pi from it.


The probability part of the assignment comes now that we have the possible values defined.


We have two variables, 0<=x<=d/2 and 0<=teta<=pi/2.

All the probability is on the square of size d/2 * pi/2, thats our space of probabilities (first time working with that?). This is the geometry of the probability.

The total area is then  d/2 * pi/2 = (pi*d) / 4, which is the total probability. (what is it?, be clearer)


$$
    A_{\text{total}} = \frac{d} {2} \frac{\pi} {2} \quad \text{(Still learningg some Latex in markdown,that's actually cool)}
$$

The needle crosses a line when x<=L/2sin(teta), hence, we can integrate the analytical function to get the probability of crossing a line. **This is the key step**.

L/2 sin teta is the x-extension of the needle, and it is at most L/2 when sin teta is 1 or teta is 0.
$$
A_{\text{cross}} = \int_0^{\pi/2} \frac{L}{2}\sin(\theta) \,d\theta.
$$

basically, since the crossing happens on 

$$ 0 \leq x \leq \frac{L} {2} \sin\theta $$

so, for eaech value of teta, we must get the probablity thatt is is smaller than  L/2 sin teta and the probability of crossing becomes 


$$
    P_\text{cross} = \frac{A_{\text{cross}}}{A_{\text{total}}}
$$

Since

$$
\int_0^{\pi/2}\sin\theta\,d\theta = 1,
$$

we have

$$
A_{\text{cross}} = \frac{L}{2}.
$$

Therefore,

$$
P_\text{cross}
=
\frac{A_{\text{cross}}}{A_{\text{total}}}.
$$

and

$$
P_\text{cross}
=
\frac{L/2}{\pi d/4} = \frac {2L} {\pi d}
$$

and this is the relation we need, we need to get $\pi$ from this:

$$
\pi = \frac {2L} {d}\frac {1} {P_{\text{cross}}}
$$


Now, comes the computation part. Now that we know the probability of the tosses, we can do an inverse problem, find $\pi$ from the defined L and d and the obtained $P_\text{cross}$. Our task becomes defining this $P_\text{cross}$.


To define $P_\text{cross}$ we need to simulate many tosses. Since every needle throw/toss is random and independent of each other we can very easily pararelize it, but lets focus first on what a toss means, lets execvute it many times, then we can think of all the different ways we can paralelize this task.


We are going to get the value from that $P_\text{cross}$ since we can find it by sampling, thats the Monte Carlo.

Hence,  the " in this case of uniform distribution gives us: 


$$ P_\text{cross} = \frac {C} {N} $$
$$ \qquad C=\text{Count of crosses} \quad \text{and} \quad N=\text{Total number of tosses}$$

This way, $\pi$ can be approximated by

$$\pi \approx \frac {2L} {d} \frac {N} {C}$$

and in the case $L=d$, we simplify to get:

$$\pi \approx \frac {2N} {C}$$

## What is a toss simulation?

We need to consider that the distance x from the nearest line is random as well as the angle the needs falls to, so, we have probability 1/(d/2) for x and probability 1/(pi/2) for teta, hence, we can simply assume a uniform distribution for both parameters, generate themn, then run the test. If it passes we return one, else, 0. simple like that.

Lets go for it. 

We will want to go pure Python -> Numpy -> Torch -> Whatever else I find for HPC/RNG/PRNG,etc...


              RNG
               │
        ┌──────┴──────┐
        ▼             ▼
        U₁             U₂
        │             │
        ▼             ▼
    d/2 × U₁        π/2 × U₂
        │             │
        ▼             ▼
        x           theta
        \           /
            \         /
            ▼       ▼
            crossing?
                │
                ▼
            counter



---- August 14

Now that I got it working, I should stop alucinating and focusing on some deliverable. Make it a git repo and define the delivarable:

Requirements per the task EP1:
Input: Number of tosses to be simulated.
Output: Pi estimate and the respective error.

But then he defines:
1. Use colab
2. Graph showing experimental error compared to the model (i/sqrt(N)) and do some analysis on the graph
3. Estimate via the graph what N is necessary to calculate pi with 14 floating point precision and how long it would take.

It seems that the basic input and output defined is only what I need for a given function to do.
The analysis is something further than that. Should I work with ipynb then? this would make this exploratory work easier and the 'program' focused on the computation. I write a library that does the work, like, cpu.py, and a EP1.ipynb to use this library to get the exploratory work done.

I should neverthless have a nice main.py that does most of the exploratory work by itself and use the ipynb only for presentation.

Should I really?
...
...
Actually, lets make it all ipynb independent. I can extract sth to an ipynb if I want but I might as well code the analysis as entrypoints for a main program.

I can either run just to test a run, wait with tqdm estimate and fine.
Or I can run in analysis mode, one mode that will save some progress timestamps with some metadata like time of execution value of current N value of current estimate for pi and the absolute error. 

Actually, I should, during simulation, only gather these data in some data structure, even on disk though I would keep it in memory for now, only after simulations are over should I start generating the report. So, I can run simulation raw and get the results of the simulation to N or I can run it with some data keeping so that I can later do the analysis.

Do I make it so that I can stop and an N/2 and continue to N or should the function to N itself keep track and return all the data already? I have to be careful here, what is the overhead it will add to the function? This I will solve as I code.


