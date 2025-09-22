# Decision Tree Example Dataset

## Ice Cream Preference Dataset

| Loves Popcorn | Loves Soda | Age | Loves Cool As Ice |
|---------------|------------|-----|---------------------|
| Yes           | Yes        | 7   | No                  |
| Yes           | No         | 12  | No                  |
| No            | Yes        | 18  | Yes                 |
| No            | Yes        | 35  | Yes                 |
| Yes           | Yes        | 38  | Yes                 |
| Yes           | No         | 50  | No                  |
| No            | No         | 83  | No                  |


## Dataset Summary
- **Total samples**: 7
- **Features**: 3 (Loves Popcorn, Loves Soda, Age)
- **Target variable**: Loves Cool As Ice (Yes/No)
- **Class distribution**: 
  - Yes: 3 samples (42.9%)
  - No: 4 samples (57.1%)

---

## Decision Tree Construction with Gini Impurity Calculations

### Step 1: Calculate Initial Gini Impurity

**Root Node (All 7 samples):**
- Yes: 3 samples, No: 4 samples
- $p_{Yes} = \frac{3}{7} = 0.429$, $p_{No} = \frac{4}{7} = 0.571$

**Gini Impurity:**
$$Gini_{root} = 1 - (p_{Yes}^2 + p_{No}^2) = 1 - (0.429^2 + 0.571^2) = 1 - (0.184 + 0.326) = 0.490$$

---

### Step 2: Evaluate All Possible Splits

#### 2.1 Split on "Loves Popcorn"


**Left Branch (Loves Popcorn = Yes): 4 samples**
- Samples: [7,No], [12,No], [38,Yes], [50,No]
- Yes: 1, No: 3
- $Gini_{left} = 1 - (\frac{1}{4})^2 - (\frac{3}{4})^2 = 1 - 0.0625 - 0.5625 = 0.375$

**Right Branch (Loves Popcorn = No): 3 samples**
- Samples: [18,Yes], [35,Yes], [83,No]
- Yes: 2, No: 1
- $Gini_{right} = 1 - (\frac{2}{3})^2 - (\frac{1}{3})^2 = 1 - 0.444 - 0.111 = 0.445$

**Weighted Gini:**
$$Gini_{weighted} = \frac{4}{7} \times 0.375 + \frac{3}{7} \times 0.445 = 0.214 + 0.191 = 0.405$$

**Information Gain:**
$$IG = 0.490 - 0.405 = 0.085$$

```
                (Root Node)
                Loves Popcorn?
                /          \
             Yes/          \No
              /              \
      (4 samples)        (3 samples)
    Yes:1, No:3          Yes:2, No:1
    Gini: 0.375          Gini: 0.445                  

```

#### 2.2 Split on "Loves Soda"

**Left Branch (Loves Soda = Yes): 4 samples**
- Samples: [7,No], [18,Yes], [35,Yes], [38,Yes]
- Yes: 3, No: 1
- $Gini_{left} = 1 - (\frac{3}{4})^2 - (\frac{1}{4})^2 = 1 - 0.5625 - 0.0625 = 0.375$

**Right Branch (Loves Soda = No): 3 samples**
- Samples: [12,No], [50,No], [83,No]
- Yes: 0, No: 3
- $Gini_{right} = 1 - (\frac{0}{3})^2 - (\frac{3}{3})^2 = 1 - 0 - 1 = 0.000$

**Weighted Gini:**
$$Gini_{weighted} = \frac{4}{7} \times 0.375 + \frac{3}{7} \times 0.000 = 0.214 + 0 = 0.214$$

**Information Gain:**
$$IG = 0.490 - 0.214 = 0.276$$


```
                (Root Node)
                Loves Soda?
                /          \
             Yes/          \No
              /              \
      (4 samples)        (3 samples)
    Yes:3, No:1          Yes:0, No:3
    Gini: 0.375          Gini: 0.000                  

```

#### 2.3 Split on "Age" (trying different thresholds)

**Split at Age ≤ 12:**


**Left Branch (Age ≤ 12): 2 samples**
- Samples: [7,No], [12,No]
- Yes: 0, No: 2
- $Gini_{left} = 1 - 0^2 - 1^2 = 0.000$

**Right Branch (Age > 12): 5 samples**
- Samples: [18,Yes], [35,Yes], [38,Yes], [50,No], [83,No]
- Yes: 3, No: 2
- $Gini_{right} = 1 - (\frac{3}{5})^2 - (\frac{2}{5})^2 = 1 - 0.36 - 0.16 = 0.480$

**Weighted Gini:**
$$Gini_{weighted} = \frac{2}{7} \times 0.000 + \frac{5}{7} \times 0.480 = 0 + 0.343 = 0.343$$

**Information Gain:**
$$IG = 0.490 - 0.343 = 0.147$$


```
                (Root Node)
                 Age <= 12 ?
                /          \
             Yes/          \No
              /              \
      (4 samples)        (3 samples)
    Yes:0, No:2          Yes:3, No:2
    Gini: 0.000          Gini: 0.408                 

```


**Split at Age ≤ 38:**

**Left Branch (Age ≤ 38): 5 samples**
- Samples: [7,No], [12,No], [18,Yes], [35,Yes], [38,Yes]
- Yes: 3, No: 2
- $Gini_{left} = 1 - (\frac{3}{5})^2 - (\frac{2}{5})^2 = 0.480$

**Right Branch (Age > 38): 2 samples**
- Samples: [50,No], [83,No]
- Yes: 0, No: 2
- $Gini_{right} = 1 - 0^2 - 1^2 = 0.000$

**Weighted Gini:**
$$Gini_{weighted} = \frac{5}{7} \times 0.480 + \frac{2}{7} \times 0.000 = 0.343 + 0 = 0.343$$

**Information Gain:**
$$IG = 0.490 - 0.343 = 0.147$$

```
                (Root Node)
                 Age <= 38 ?
                /          \
             Yes/          \No
              /              \
      (4 samples)        (3 samples)
    Yes:3, No:2          Yes:0, No:2
    Gini: 0.480          Gini: 0.000                 

```

---

### Step 3: Choose Best Split

**Comparison of Information Gains:**
- Loves Popcorn: IG = 0.085
- **Loves Soda: IG = 0.276** ⭐ **BEST**
- Age ≤ 12: IG = 0.147
- Age ≤ 38: IG = 0.147

**Winner: Split on "Loves Soda" with highest Information Gain of 0.276**

### Step 4:  Decision Tree Visualization

```
                    Root Node
                   (7 samples)
                Yes:3, No:4, Gini:0.490
                      |
                Loves Soda?
                /          \
             Yes/          \No
              /              \
      (4 samples)        (3 samples)
    Yes:3, No:1          Yes:0, No:3
    Gini: 0.375          Gini: 0.000
         |                    |
     (NEW ROOT)       PREDICT: No
                          (Pure node)


```
**Next Step**
Repeat the whole process again for the new root node:
  - Calculate the gini 
  - Choose the lowest value and start splitting 