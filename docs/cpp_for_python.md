# C/C++ in EP4, for a Python programmer

What is new in the EP4 engine if you come from Python: how the files are organized, why headers contain whole functions, and the syntax that has no Python equivalent.

## 1. How a C++ program is built (vs `import`)

**Python:** `import geometry` runs `geometry.py` once, creates a module object, and every file that imports it shares that object. Names are looked up while the program runs.

**C/C++:** there are no modules at run time. Building happens in three stages before anything runs:

```
            preprocessor           compiler              linker
main.cu  ──(paste #includes)──▶  main.o   ─┐
scene.cu ──(paste #includes)──▶  scene.o  ─┼──(join, resolve names)──▶  star (one executable)
image.cu ──(paste #includes)──▶  image.o  ─┤
options.cu ─(paste #includes)─▶  options.o ┘
```

1. **Preprocessor.** `#include "rng.cuh"` is **text copy-paste**: the whole content of `rng.cuh` is pasted where the line was. No module, no namespace, no "run once". `#define`, `#ifdef` are also text operations done here.
2. **Compiler.** Each `.cu` file, *after* pasting, is a **translation unit**, compiled alone into an object file (`.o`) of machine code. When compiling `main.cu`, the compiler knows nothing about `scene.cu`, only what was pasted into `main.cu`.
3. **Linker.** Joins the object files. When `main.o` calls `build_scene`, the linker finds its machine code in `scene.o` and connects the call.

One `nvcc` call does all three: `nvcc main.cu options.cu scene.cu image.cu -o star` (phase B, from a Makefile). Since 2026-11-21 there is no Makefile: `engine.py` composes that call itself, with the sources of each binary ([gpu_engine.md §3](gpu_engine.md#3-building-from-python-no-makefile-decided-2026-11-21)).

## 2. Declaration vs definition

The compiler must know a function's **signature** before you call it, but not necessarily its **body**.

```c
int build_scene(const StarConfig *config, Scene *scene);      // declaration: "this exists somewhere"

int build_scene(const StarConfig *config, Scene *scene) {     // definition: the actual code
    ...
}
```

- **The classic use of a header** is a list of declarations: `scene.cuh` *declares* `build_scene`, `scene.cu` *defines* it. `main.cu` includes `scene.cuh`, so it can call `build_scene`, and the linker connects the call to the code in `scene.o`.
- **The one-definition rule.** A normal (non-`static`, non-`inline`) function must be **defined exactly once** in the whole program.
  - If two `.cu` files included a header that *defined* a normal function, the linker would find two copies and fail with "multiple definition".
  - That is why classic headers only declare.

## 3. Why our headers contain whole functions

Some of our headers (`real.cuh`, `rng.cuh`, `geometry.cuh`, `photon.cuh`) hold complete functions, marked `static inline`. That is a deliberate choice, for three reasons:

1. **Speed in the hot loop.**
   - To *inline* a function (paste its body into the caller instead of jumping to it), the compiler must **see the body** while compiling the caller.
   - `trace_photon` runs 10⁸–10¹² times. Its helpers (`uniform`, `sine_from_cosine`, `bin_index`, ...) are tiny, and a function call would cost about as much as their work.
   - With the bodies in headers pasted into `main.cu`, the compiler can inline the whole photon into the loop.
2. **The GPU.**
   - By default `nvcc` compiles device code **one translation unit at a time**, with no device linker: a kernel in `main.cu` can only call `__device__` functions whose bodies are in `main.cu`, pasted or written there.
   - Separate device linking exists (`-rdc=true`), but it is slower to build and inlines less.
   - Header-only physics makes the future kernel work with no extra flags.
3. **Templates** (section 5) must be in headers: the compiler generates code from them when they are used, so it needs the body at that point.

And the two keywords that make it legal:

- **`inline`** tells the linker that identical copies of this function in several translation units are fine; keep one. It is also a hint to inline, which modern compilers decide on their own anyway.
- **`static`** (on a function) means **visible only inside this translation unit**: each `.cu` that includes the header gets its own private copy, and the copies never clash at link time.
  - It is the closest thing to Python's leading underscore `_helper`, except that it is enforced.
  - In `scene.cu`, `static` marks the helpers (`random_point_on_sphere`, `place_spots`, ...) as private to that file; only `build_scene` is public.

## 4. How EP4 is organized

| File | Holds | Why there |
|---|---|---|
| `real.cuh` | `typedef real`, macros, `real_*` math, `sine_from_cosine` | used everywhere, tiny, hot: header-only |
| `rng.cuh` | Philox types and functions | hot (called ~6 times per photon): header-only |
| `geometry.cuh` | `Vec3`, `distance_squared` | hot: header-only |
| `photon.cuh` | `trace_photon` and its steps (templates) | the hot loop itself, and the future GPU code: header-only |
| `scene.cuh` + `scene.cu` | types in the header; `build_scene` declared in the header, defined in the `.cu` | runs once per run: a normal function, compiled once; helpers hidden with `static` |
| `image.cuh` + `image.cu` | `Image` type, declarations; only `count_photon` inline in the header (called per photon) | allocate, save and free run once; `count_photon` is hot |
| `options.cuh` + `options.cu` | `Options` type, `load_options` declared; parsing details hidden in the `.cu` | runs once; `main` does not need to know how parsing works |
| `main.cu` | `main`, the chunk loop, the result line | includes everything, the only file with `main` |

**The rule we follow:** *hot path and GPU-bound code → whole functions in headers (`static inline`); code that runs once → declared in the header, defined in a `.cu`.*

### Include guards

Every header starts and ends like this:

```c
#ifndef RNG_CUH
#define RNG_CUH
...
#endif
```

`photon.cuh` includes `rng.cuh`, and `main.cu` includes both. Without the guard, `rng.cuh` would be pasted twice into `main.cu`, and the second copy of every `struct` would be an error. The first paste defines `RNG_CUH`; the second time, `#ifndef RNG_CUH` is false and the content is skipped. It does by hand what Python's "a module is imported only once" does automatically.

## 5. Syntax that is new from Python

| C/C++ | What it means | Python analogy |
|---|---|---|
| `#define TWO_PI ((real) 6.28...)` | text substitution before compiling, no type, no scope | a constant, but done by find-and-replace |
| `#ifdef STAR_FP32 ... #else ... #endif` | keep or drop code **at compile time**: the `star_cpu32` build passes `-DSTAR_FP32` | `if` at import time, but the dropped code does not even exist in the binary |
| `typedef double real;` | a new name for a type | `real = float` (a type alias) |
| `typedef struct { real x, y, z; } Vec3;` | a record of fields, laid out contiguously in memory, no methods | a `@dataclass` without methods |
| `Vec3 p = {1, 2, 3};` | initialize fields in order | `Vec3(1, 2, 3)` |
| `p.x` vs `scene->nx` | field of a value vs field through a pointer (`a->b` is `(*a).b`) | always `obj.attr` in Python |
| `Scene *scene` | a **pointer**: the memory address of a `Scene` | every Python variable is a reference; C makes it explicit |
| `&rng` | "the address of `rng`": lets a function modify the caller's variable | passing a mutable object |
| `const Scene *scene` | the function promises not to modify what it points to | no equivalent (convention only) |
| `unsigned int`, `long long`, `unsigned long long` | fixed-size integers: 32 bits without sign, 64 bits, 64 bits without sign | Python `int` is unbounded |
| `(unsigned int) x`, `(real) 0.6` | a cast: convert to another type (may truncate) | `int(x)`, `float(x)` |
| `x++`, `counts->recorded++` | add 1 | `x += 1` |
| `a ? b : c` | conditional expression | `b if a else c` |
| `static inline HD float real_sqrt(float x)` and `...double real_sqrt(double x)` | **overloading**: same name, different parameter types; the compiler picks by type | no equivalent (Python would check the type at run time) |
| `template <typename Rng> ... trace_photon(const Scene *scene, Rng *rng)` | a **template**: the function is written for "some type `Rng`"; the compiler generates one version per type it is used with (today `PhiloxState`, later cuRAND's state) | duck typing, but resolved at compile time: a missing `uniform(Rng*)` is a compile error, not a run-time one |
| `switch (x) { case 0: ...; break; }` | jump to the matching case | `match x: case 0: ...` |
| `enum { OPT_SPOTS = 1000, ... }` | named integer constants | `enum.IntEnum` |
| `do { ... } while (condition);` | loop that runs the body at least once | `while True: ...; if not condition: break` |

### Two habits behind these choices

- **Memory is explicit.** `malloc` asks for bytes and `free` gives them back; nothing is garbage-collected. In CUDA the same idea applies to GPU memory (`cudaMalloc`, `cudaFree`).
- **Types decide speed.** Python decides types while running; C++ fixes them when compiling. That is how `real_sqrt(float)` becomes one single hardware instruction, and how `#ifdef STAR_FP32` builds a different binary from the same source.

## 6. Putting it together: guards, `static inline`, templates and `HD`

The most loaded line in EP4 is the start of `trace_photon`:

```c
template <typename Rng>
static inline HD long long trace_photon(const Scene *scene, Rng *rng)
```

Word by word:

| Piece | Who reads it | What it does |
|---|---|---|
| `template <typename Rng>` | the C++ compiler | the function is a recipe for "some generator type `Rng`"; real code is generated per type, when used |
| `static` | the linker | this copy is private to the `.cu` that pasted it: copies in other files never clash |
| `inline` | the compiler and the linker | "identical copies are fine", and a hint to paste the body into the caller |
| `HD` → `__host__ __device__` | `nvcc` | compile the function for the CPU **and** for the GPU |
| `long long` | everyone | the return type: a pixel index, or a negative code (`PHOTON_BLOCKED_BY_SPOT`, ...) |
| `const Scene *scene` | the compiler | a pointer to the scene; `const` promises not to change it |
| `Rng *rng` | the compiler | a pointer to the generator: drawing a number advances it, so the caller's generator must change |

### 6.1 Include guards work per translation unit

```c
#ifndef PHOTON_CUH
#define PHOTON_CUH
...
#endif
```

- **What the guard protects.** It stops a header from being pasted **twice into the same `.cu`**, for example `rng.cuh` reaching `main.cu` directly and again through `photon.cuh`. Without it, the second copy of every `struct` is a compile error ("redefinition").
- **What it does not protect.** Each `.cu` is compiled alone, from scratch. `main.cu` and `scene.cu` *each* get one copy of `rng.cuh`: the guard does nothing between files, because `RNG_CUH` is defined only while compiling one file.
- **That is why the functions also need `static` / `inline`.** Each `.cu` ends up with its own copy of `philox_next`, and those keywords make those copies legal at link time.
- **`#pragma once`** does the same as a guard in one line. Every major compiler supports it, but it is not standard C++; EP4 uses guards, like the CUDA headers.

### 6.2 `static inline`: one copy per file, ideally zero calls

- **The goal** is that the photon loop in `main.cu` contains no calls at all: `uniform`, `sine_from_cosine`, `bin_index`, ... pasted into it, so the compiler can optimize the whole photon as one block (keep values in registers, reorder, fuse multiply-adds).
- **Did it work?** With `-O2`, the compiler inlines small functions it can see. That is why a 50-line `trace_photon` split into nine small functions costs nothing compared with writing it as one long function: we get readable code and fast code.
- **`static` alone or `inline` alone would also link here.** Using both is the common C idiom for "helper defined in a header": private to each file, and inlinable.

### 6.3 `HD`: one source, two compilations

`nvcc` compiles a `.cu` file **twice**:

```
          ┌─ host pass   → gcc  → CPU machine code   (__host__ functions)
main.cu ──┤
          └─ device pass → NVVM → PTX → SASS for -arch   (__device__ and __global__ functions)
```

- **`__host__ __device__`** functions go through both passes: one source, two machine codes.
- **Two macros tell the passes apart:**
  - `__CUDACC__` is defined in **both** passes, whenever `nvcc` compiles. Our `HD` macro checks it, so `HD` becomes `__host__ __device__` under `nvcc` and disappears under plain `gcc`.
  - `__CUDA_ARCH__` is defined **only in the device pass**, with the target architecture (1200 for the laptop, 800 for the A30).
  - Code can branch on it to do the same job differently on each side. cuRAND's own `mulhilo32` does exactly that: a 64-bit multiply on the host, the `__umulhi` instruction on the device.
- **Today** there is no kernel, so nothing calls the physics from device code: the device pass generates nothing for it, and the build is effectively a CPU build. As soon as a kernel calls `trace_photon`, the device pass compiles the same functions for the GPU.

### 6.4 Templates + `HD`: why `trace_photon` is a template

`trace_photon` must work with two different generators:

- **On the CPU:** our `PhiloxState`, with `uniform(PhiloxState *)` from `rng.cuh`. It is `HD`, so it works on both sides.
- **On the GPU:** cuRAND's `curandStatePhilox4_32_10_t`, with a `uniform(curandStatePhilox4_32_10_t *)` that calls `curand_uniform_double`. That one is **device-only**: we measured that cuRAND does not compile in host code.

**If `trace_photon` were a plain `HD` function** taking cuRAND's state, its host version would call a device-only function, and `nvcc` would refuse it.

**As a template**, nothing is generated until it is used, and only for the combination that is used:

| Instantiation | Where it is called from | Generated for |
|---|---|---|
| `trace_photon<PhiloxState>` | `simulate_chunk` (CPU) | host |
| `trace_photon<curandStatePhilox4_32_10_t>` | the future kernel (GPU) | device |

So the template gives one source of physics, generated for CPU with our Philox and for GPU with the real cuRAND, and each version only calls what exists on its side. This is the main reason `trace_photon` and its random-number helpers are templates and not plain functions. It is compile-time duck typing:

- whatever `Rng` is, it needs a `uniform(Rng *)` function returning a `real`;
- if it is missing, the build fails with an error, instead of a crash at run time.

### 6.5 Where to read more

- **CUDA C++ Programming Guide:** "function execution space specifiers" (`__host__`, `__device__`, `__global__`), "`__CUDA_ARCH__`", "separate compilation" (`-rdc`), and the C++ language support section (templates in device code).
- **The `nvcc` documentation:** "the CUDA compilation trajectory" (the host and device passes).
- **Any C++ reference:** "one definition rule", "inline specifier", "internal linkage" (`static`), "templates: implicit instantiation".
