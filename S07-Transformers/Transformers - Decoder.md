# Transformers - Decoder
![transformer](images/transformer.png)

The **decoder** is used in **sequence generation tasks** (translation, summarization, etc.). Unlike the encoder, it **not only looks at the input sequence** (from the encoder) but also **the tokens generated so far**.

A Transformer decoder block has **3 main components**:

1. **Masked Multi-Head Self-Attention**
2. **Encoder-Decoder Cross-Attention**
3. **Feed-Forward Network (FFN)**
4. **Residual connections + LayerNorm** after each

---

## 1. Why Masked Self-Attention?

Imagine we are generating a translation:

```
Input:   "I am a student"
Output:  "Je suis un..."
```

At time step `t` in the decoder:

* We **cannot see future tokens** (e.g., the word after “un”) because it would be cheating.
* **Masked self-attention** ensures each token can only attend to **earlier or current tokens**, not future ones.

---

### 1.1 Masking in Action

Suppose we have 4 generated tokens: `[Je, suis, un, étudiant]`

The attention mask looks like this:

| Query \ Key | Je | suis | un | étudiant |
| ----------- | -- | ---- | -- | -------- |
| Je          | 1  | 0    | 0  | 0        |
| suis        | 1  | 1    | 0  | 0        |
| un          | 1  | 1    | 1  | 0        |
| étudiant    | 1  | 1    | 1  | 1        |

* `1` → allowed to attend
* `0` → masked

This prevents **future token leakage**.
Imagine like multiplying the dot product matrix by this to ensure that softmax is made only on present or past tokens.

---

## 2. Encoder-Decoder Cross Attention

After masked self-attention, the decoder has **cross-attention**.

This is the key difference from the encoder.

### 2.1 Intuition

* The **decoder needs to look at the encoder output** to know **what information from the input sequence is relevant**.
* Example:

```
Input: "I am a student"
Decoder so far: "Je suis un"
```

* Cross-attention allows "un" to look at the encoder’s representations of **“I am a student”** to predict the next French word **“étudiant”** correctly.

---

### 2.2 Inputs

Let:

* **Decoder hidden states after masked self-attention** = `H_dec` (shape `[t, d_model]`)
* **Encoder output** = `H_enc` (shape `[n, d_model]`)

We compute **queries from decoder** and **keys & values from encoder**:

$$
Q = H_{dec} W^Q_{cross}, \quad K = H_{enc} W^K_{cross}, \quad V = H_{enc} W^V_{cross}
$$

* Shapes:

Let’s define dimensions explicitly:

| Matrix | Meaning      | Shape                                           |
| ------ | ------------ | ----------------------------------------------- |
| Q      | Query matrix | `[seq_len, d_k]` (sometimes written `[t, d_k]`) |
| K      | Key matrix   | `[seq_len, d_k]` (sometimes `[n, d_k]`)         |
| V      | Value matrix | `[seq_len, d_v]`                                |

* $d_k$ = dimension of **queries and keys**
* $d_v$ = dimension of **values**
* $d_{model}$ = embedding size of the model

> So yes, in some books you might see `d_Q` instead of `d_k`. They often assume `d_Q = d_K` because the dot-product attention requires the query and key dimensions to match.



---

### 2.3 Attention Computation

Cross-attention is **normal attention**, just Q from decoder, K/V from encoder:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

* Output shape = `[t, d_model]`
* Each decoder token attends to **all encoder tokens**.

---

### 2.4 Example

Input: `"I am a student"` → encoder output `[H_I, H_am, H_a, H_student]`

Decoder token: `"un"`

* Query `Q_un` interacts with **all encoder keys** `[K_I, K_am, K_a, K_student]`
* Softmax gives attention weights → weighted sum of **encoder values**
* Output = context-aware representation of `"un"` based on input

---

### 2.5 Multi-Head Cross-Attention

* Same principle as self-attention
* Multiple heads allow the decoder token to attend to **different aspects of the input sequence**
* Example: one head focuses on nouns, another on verbs, another on sentence structure



#### 1. What multi-head attention really does

* Each **head has its own set of Q, K, V projection matrices**.
* Each head produces an **independent attention output**.
* Intuition: each head can “focus” on **different kinds of patterns** in the data (e.g., syntax, semantics, long-range dependencies).

> Important: **the heads are *not hard-coded* to nouns, verbs, or anything specific**. They just learn whatever patterns are useful for the task during training.

---

#### 2. Cross-attention in the decoder

* In cross-attention, the **query comes from the decoder** (the token being generated)
* The **key and value come from the encoder** (the input sequence)

Each decoder token asks:

> “Which parts of the input sequence should I pay attention to right now?”

* Each **head produces its own weighted sum of the encoder values**.
* Then, the outputs of all heads are **concatenated** and projected through a linear layer → **final vector for this decoder token**

So, it’s **not that one head handles verbs and another nouns explicitly**, instead:

1. **All heads are always active.**
2. Each head may, during training, specialize to capture certain relationships (sometimes one focuses more on verbs, another on nouns, etc.).
3. **The outputs of all heads are combined (concatenated and projected).**

✅ **Analogy:**
Think of each head as a “lens” or “perspective”:

* Head 1 sees **subject-verb relationships**
* Head 2 sees **object-verb relationships**
* Head 3 sees **long-range dependencies**

You **don’t choose a head manually**, the model learns during training how to distribute attention across heads. The **decoder token gets the final representation after combining all heads**, which contains **all these perspectives together**.

---

#### 3. TL;DR

* **No single head handles only verbs or nouns.**
* **All heads run in parallel**, each learns a different type of relationship.
* **Their outputs are concatenated** → one final embedding for the token.
* Specialization happens automatically during training, but the model uses all heads for **every token**.




---

## 3. Decoder Block Summary

For each decoder block:

1. **Masked Multi-Head Self-Attention**

   * Input: decoder embeddings so far
   * Output: updated token embeddings, aware of past tokens

2. **Residual + LayerNorm**

   * `x = LayerNorm(x + MaskedSelfAttention(x))`

3. **Encoder-Decoder Cross Attention**

   * Query = decoder embeddings
   * Key/Value = encoder outputs
   * Output = embeddings enriched with **encoder info**

4. **Residual + LayerNorm**

   * `x = LayerNorm(x + CrossAttention(x, H_enc))`

5. **Feed-Forward Network (FFN)**

   * Position-wise FFN like encoder
   * Residual + LayerNorm

---


## 4. Full Decoder Flow Example (Translation)

Input: `"I am a student"` → Encoder output `[H1, H2, H3, H4]`

Decoder generates: `"Je suis un étudiant"`

1. Step 1: Decoder token `"Je"`
   * Masked self-attention → attends only to `"Je"` (first token)
   * Cross-attention → attends to all encoder outputs `[H1-H4]`
   * FFN → transforms embedding
2. Step 2: Decoder token `"suis"`
   * Masked self-attention → attends to `"Je"` and `"suis"`
   * Cross-attention → attends to encoder outputs `[H1-H4]`
   * FFN → transforms embedding
3. Repeat until end-of-sequence token

---

## 5. Training vs Inference in Decoders

### 5.1 Training

* We **already know the target output** (supervised learning).
* Example: translating `"I am a student"` → `"Je suis un étudiant"`

**Decoder input:**

```
<BOS> Je suis un
```

* We feed the **entire target sequence shifted by one**.
* At each time step, the decoder **sees the correct previous token**:

| Decoder step | Input token | Target token |
| ------------ | ----------- | ------------ |
| 1            | `<BOS>`     | Je           |
| 2            | Je          | suis         |
| 3            | Je suis     | un           |
| 4            | Je suis un  | étudiant     |

* This is called **teacher forcing**.
* **Masked self-attention** still applies, so each token sees only **previous tokens**.
* **Why?** Makes training faster and more stable because the model **always conditions on the correct previous token**, not its own possibly wrong prediction.
* Cross-attention works the same: decoder queries attend to encoder outputs.

---

### 5.2 **Inference (generation)**

* We **don’t know the target**. We generate it step by step.

**Process:**
1. Start with `<BOS>`
2. Masked self-attention: `<BOS>` attends only to itself
3. Cross-attention: attends to encoder outputs
4. Predict **first token** → `"Je"`
5. Next step: decoder input = `<BOS>` + `"Je"`
6. Masked self-attention:
   * `<BOS>` attends to itself
   * `"Je"` attends to `<BOS>` and itself
7. Cross-attention: attends to encoder outputs
8. Predict **next token** → `"suis"`

**Repeat** until end-of-sequence token is predicted.

---

### Key Differences

| Aspect             | Training                                  | Inference (Decoding)             |
| ------------------ | ----------------------------------------- | -------------------------------- |
| Previous token     | Correct token from ground truth           | Previously **predicted token**   |
| Teacher forcing    | ✅ Yes                                     | ❌ No                             |
| Masked attention   | Same                                      | Same                             |
| Speed              | Fast (can process whole sequence at once) | Slower (step-by-step generation) |
| Error accumulation | Low (correct tokens given)                | High (mistakes compound)         |

---

### 5.3 Example (step-by-step)

Input: `"I am a student"` → Encoder output `[H1,H2,H3,H4]`

**Training:**

* Step 1: `<BOS>` → `"Je"` ✅
* Step 2: `"Je"` → `"suis"` ✅
* Step 3: `"Je suis"` → `"un"` ✅

**Inference:**

1. **Step 1: `<BOS>`**
   * Decoder masked self-attention sees only `<BOS>`
   * Cross-attention attends to `[H1, H2, H3, H4]`
   * Model predicts **"Je"** ✅ (correct)

2. **Step 2: `<BOS> Je`**
   * Masked self-attention:
     * `<BOS>` attends to itself
     * `"Je"` attends to `<BOS>` and itself
   * Cross-attention attends to `[H1,H2,H3,H4]`
   * Suppose model makes a **mistake** and predicts **"est"** instead of `"suis"` ❌

3. **Step 3: `<BOS> Je est`**
   * Masked self-attention:
     * `"Je"` attends to `<BOS>` and itself
     * `"est"` attends to `<BOS> Je est` (masked)
   * Cross-attention attends to `[H1,H2,H3,H4]`
   * Because previous token `"est"` is wrong, the model now has **wrong context**
   * Model predicts `"un"` ❌ (or maybe `"le"`), not `"un"`

4. **Step 4: `<BOS> Je est un`**
   * Final token `"étudiant"` is likely **not predicted**, because the model was misled by earlier mistakes.

> Notice: during inference, each prediction **depends on previous predictions**, so mistakes can propagate.



---


### 5.4 Key insight

* **Training:** The model always sees the correct previous token → `"Je suis un"` → `"étudiant"` is predicted correctly.
* **Inference:** Each predicted token becomes the next token's input → mistakes compound.
* **Effect:** Even one wrong token early on (like `"est"` instead of `"suis"`) can prevent the correct output `"étudiant"` from ever being predicted.

> This is why **inference is harder than training** and why techniques like [**beam search**](https://www.geeksforgeeks.org/machine-learning/introduction-to-beam-search-algorithm/) are used: they keep multiple candidate sequences to reduce error propagation.




---

## 6. The Transformer Problem: Quadratic Attention

Standard Transformers (like GPT, BERT, vanilla encoder-decoder) use **self-attention**:

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

* For a sequence of length `n`, the **attention matrix `QK^T` has shape `[n,n]`**.
* This means **each token attends to every other token** in the sequence.

**Problem:**

* Memory usage: (O(n^2))
* Computation: (O(n^2))

✅ Works fine for short sequences (512–1024 tokens)
❌ Becomes **impossible for long sequences** (like 10k, 100k tokens).

Example: a book or long article → vanilla Transformers can’t handle it efficiently.

---

### 6.1 Why Long Sequences are Hard

1. **Quadratic memory**: storing the attention matrix blows up with long `n`.
2. **Quadratic compute**: computing all `n^2` dot products is slow.
3. **Context limitation**: even if you increase `n`, standard Transformers are limited by GPU memory and efficiency.

This is a **major bottleneck** in tasks like:

* Document summarization
* Long-form Q&A
* Code or DNA sequence modeling

---

### 6.2 Solutions: Sparse & Efficient Attention

Researchers designed **variants of Transformers** to handle long sequences:

#### 6.2.3 Longformer

* Introduces **sparse attention**:
  * Each token attends only to a **local window** of tokens (`w` tokens before & after).
  * Some **global tokens** can attend to all positions (for classification or summarization).
* Complexity reduced from **O(n²)** → **O(n·w)**, where `w << n`.
* Handles sequences of **thousands of tokens** efficiently.

Good for long documents without losing too much context.

---

#### 6.2.4 Other Variants

| Model         | Idea                                                                                 |
| ------------- | ------------------------------------------------------------------------------------ |
| **Reformer**  | Use **LSH attention** to attend only to similar tokens → reduces memory and compute. |
| **Performer** | Approximate softmax with **kernel methods** → linear attention.                      |
| **Linformer** | Project keys/values to a **lower dimension** → linear memory.                        |
| **BigBird**   | Combines **local + global + random attention** → can handle really long sequences.   |

---

### 6.3 Key Takeaways

* **Vanilla Transformer**: great for short sequences, inefficient for long text.
* **Problem**: quadratic memory & compute → can’t scale to long contexts.
* **Longformer / BigBird / Performer etc.**: use **sparse or approximated attention** → handle much longer sequences efficiently.
* **Effect**: you can process entire books, long conversations, or long code files without running out of GPU memory.

---

💡 **Analogy:**

* Vanilla Transformer = “everybody in a room talks to everybody else” → chaos & expensive.
* Longformer = “each person talks only to neighbors + a few VIPs” → much more efficient but still shares important info.




---
## 7. Transformers - Autoregressive Decoder (e.g., GPT)

Unlike encoder-decoder models (like translation with encoder output `H_enc`), **GPT-style models are purely decoders**:

* They are **autoregressive**, meaning **they generate one token at a time**, conditioning only on previously generated tokens.
* There is **no encoder**, everything is based on **past context**.

---

### 7.1 What Does “Autoregressive” Mean?

**Autoregressive** = “previous outputs become the next input.”

* In math:

$$
P(x_1, x_2, ..., x_T) = P(x_1) \cdot P(x_2 \mid x_1) \cdot P(x_3 \mid x_1,x_2) \cdots P(x_T \mid x_1..x_{T-1})
$$

* GPT models **factorize the sequence probability** like this.
* The model predicts **next token** based only on **all previous tokens**.
* This is exactly like **masked self-attention** in your decoder, but there’s **no cross-attention to an encoder**.

---

### 7.2 GPT Decoder Block

Each GPT decoder block is very similar to your current decoder block, **without the encoder cross-attention**:

1. **Masked Multi-Head Self-Attention**
   * Each token attends to **all previous tokens only** (mask future tokens).
   * Output: contextual embedding based on past tokens.
2. **Feed-Forward Network (FFN)**
   * Position-wise FFN enriches embeddings.
3. **Residual Connections + LayerNorm**
   * Applied after each module.

> Key difference from encoder-decoder: **No cross-attention layer**, because there is no encoder input.

---

#### 7.2.1 Masked Self-Attention in GPT

Suppose the current sequence is:

```
<I> <am> <a> 
```

* When predicting the next token (`<student>`):

  * Query = current token embedding (`<a> `)
  * Keys & Values = all previous token embeddings `[<I>, <am>, <a>]`
  * Softmax over masked dot-product → weighted sum → context-aware representation of `<student>`
  * Linear + softmax → probability for next token

All heads participate; their outputs are **concatenated** → final embedding.

---

#### 7.2.2 Generation (Autoregressive)

**Step-by-step generation:**

| Step | Input tokens   | Model predicts next token |
| ---- | -------------- | ------------------------- |
| 1    | `<BOS>`        | `I`                       |
| 2    | `<BOS> I`      | `am`                      |
| 3    | `<BOS> I am`   | `a`                       |
| 4    | `<BOS> I am a` | `student`                 |

* Only **previous tokens** are visible at each step.
* Mask ensures **future tokens are not leaked**.
* If a wrong token is predicted, the error propagates in the next step (same as encoder-decoder inference).

---

#### 7.2.3 Key Points About GPT

1. **Autoregressive** → generates tokens one at a time based on **past context**.
2. **No encoder needed** → purely self-attention over past tokens.
3. **Masked self-attention** ensures causality (no cheating with future tokens).
4. **Multi-head attention** still allows the model to focus on **different relationships** (syntax, semantics, etc.).
5. **Stacked decoder blocks** → each token embedding enriched layer by layer → final logits → softmax → next token.

---

### 7.3 Example: Sentence Completion

Suppose GPT sees:

```
"I am a"
```

1. Masked self-attention computes attention of `<I>`, `<am>`, `<a>` only to past tokens.
2. Next token predicted: `"student"`
3. Sequence becomes `"I am a student"` → next token prediction uses all four tokens as context.

> Notice: **all predictions depend on previously generated tokens** → autoregressive property.

### 7.4 GPT Translation
Yes, I'm hearing you wondering but I tried ChatGPT with translation!
GPT **does everything autoregressively**, it **predicts the next token**,but from a user’s perspective it seems like it “understands” tasks, answers questions, or even translates text. Here's how that works:

---

#### 7.4.1 Autoregressive Generation

GPT is an **autoregressive model**, meaning:

* It always generates **one token at a time**, from left to right.
* Each token prediction depends on **all previous tokens**.
* The process repeats until it predicts an **end-of-sequence token**.

So fundamentally:

```
next_token = GPT(previous_tokens)
```

No matter if the task is translation, answering questions, or summarizing.

---

#### 7.4.2 How GPT Handles Prompts

When you give GPT a prompt like:

```
Translate "I am a student" to French
```

1. GPT **tokenizes** the entire prompt:

```
["Translate", '"', "I", "am", "a", "student", '"', "to", "French"]
```

2. These tokens go through the **stack of transformer decoder blocks**. Each token embedding is updated by:

* **Masked self-attention:** each token attends to all previous tokens (in the prompt).
* **Feed-forward networks + residual connections**

3. After the prompt tokens, GPT predicts **the next token**. At this point, it doesn’t know you want a “translation” explicitly; it relies on **patterns learned from training**:

* GPT has seen countless examples where a prompt like `"Translate ... to French"` is followed by the correct translation.
* So when predicting the next token, the model “expects” the French word to come next.

---

#### 7.4.3 Step-by-Step Example

Prompt:

```
Translate "I am a student" to French:
```

Tokens:

```
[Translate, ", I, am, a, student, ", to, French, :]
```

##### Step 1: Predict first token of answer

* Model sees all the prompt tokens.
* Masked self-attention allows each token to attend to previous ones.
* Output probability distribution over next token → most likely first French word `"Je"`.

##### Step 2: Predict second token

* Input sequence now:

```
[Translate, "I, am, a, student", to, French, :, Je]
```

* Masked self-attention: `"Je"` attends to all previous tokens (the entire prompt).
* Predict next token → `"suis"`.

##### Step 3: Continue

* Input now:

```
[Translate, "I, am, a, student", to, French, :, Je, suis]
```

* Predict next token → `"un"`, then `"étudiant"`, etc.

> Each token is generated **conditioned on everything before it**, including the prompt and previous generated tokens.

---

#### 7.4.4 Why GPT Can Do Tasks Beyond “Next Word”

Even though it’s **always predicting the next token**, GPT can:

* Translate text
* Summarize
* Answer questions

**Because during training:**

* It saw many examples of **text + instruction → desired output**.
* It **learned patterns** connecting instructions with correct responses.
* So when it sees a prompt, it predicts **the most likely continuation that matches those patterns**.

So “predicting the next token” + **training on diverse instruction-text pairs** = GPT performing tasks.

---

#### 7.4.5 Key Insight
##### Comparison: Encoder-Decoder vs Autoregressive Decoder

| Feature               | Encoder-Decoder (T5, Transformer) | Autoregressive Decoder (GPT)            |
| --------------------- | --------------------------------- | --------------------------------------- |
| Cross-attention       | Yes, attends to encoder output    | No                                      |
| Masked self-attention | Yes, prevents future token leak   | Yes, same reason                        |
| Autoregressive        | Only during inference             | Always, by design                       |
| Training              | Teacher forcing                   | Teacher forcing (next-token prediction) |
| Use-case              | Translation, summarization        | Text generation, completion             |

---

✅ **TL;DR GPT / Autoregressive Decoder**

* Predicts one token at a time, based only on **previous tokens**.
* **No encoder input** is needed.
* Uses **masked multi-head self-attention** + FFN + residuals.
* Generates text step by step → errors propagate if wrong token predicted.
* **GPT doesn’t have a translation module or a math module.**
* Everything is **next-token prediction**.
* Task-solving emerges from **patterns in its training data**.
* Prompting effectively **guides GPT’s predictions**.



---


## 8. Key Differences Encoder vs Decoder

| Feature              | Encoder                 | Decoder                                             |
| -------------------- | ----------------------- | --------------------------------------------------- |
| Self-Attention       | Yes, attends all tokens | Yes, but **masked** to prevent future token access  |
| Cross-Attention      | No                      | Yes, attends encoder output                         |
| Feed-Forward Network | Yes                     | Yes                                                 |
| Residual + LayerNorm | After each block        | After each block                                    |
| Purpose              | Encode input info       | Generate output using past output + encoder context |

---

✅ **TL;DR Decoder**

1. **Masked self-attention** → see only previous tokens
2. **Cross-attention** → look at encoder outputs for context
3. **FFN + residuals + LayerNorm** → enrich each token individually
4. Stacked **N decoder blocks** → final embeddings → softmax → generate tokens


---

