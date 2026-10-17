# 🐍 Python for Machine Learning

[Open my Colab](https://colab.research.google.com/drive/1LN-6UpINT0bapOO4hxXtEydbcH_1ZGbX) · Notebook 01 of 16

> **Core idea:** Python gives us a readable way to represent data, transform it, and express the repeated computations used throughout machine learning.

## 🎯 Purpose & learning objectives

This notebook builds programming fluency before numerical libraries or models. Learn to choose a suitable collection, manipulate text and numbers, express conditions and repetition, and package reusable transformations in functions. The examples are general Python foundations rather than a trained ML model.

## 🗺️ Concept map

**Values → collections → conditions → loops → reusable functions → concise data transformations.** Each later notebook relies on these basics when it creates arrays, processes labels, or repeats optimization steps.

## 📚 Walkthrough of all major sections

| Section | What the code teaches | Why it matters |
|---|---|---|
| 1 · Variables and numbers | Assignment, `type`, integer/float arithmetic, division, floor division, modulo, powers | Names hold inputs and intermediate results; numeric operators express model calculations |
| 1 · Strings | Concatenation, equality, capitalization, case conversion, splitting and replacement | Text preprocessing starts with consistent string operations |
| 2 · Lists | Indexing, negative indices, slices, replacement, deletion, append, pop, remove, sort, reverse and nested lists | Lists preserve ordered observations and allow mutation |
| 2 · Dictionaries | Keys/values/items, membership, `get`, `setdefault`, character counting | Count categories or map labels to values without scanning a list |
| 2 · Tuples and sets | Tuple indexing; unique set members versus duplicate list members | Choose immutable records or unique collections appropriately |
| 3 · Comparisons and logic | `==`, inequalities, `and`, `or`, `not` | Build predicates for filtering and control flow |
| 4 · Control flow | `if`/`elif`/`else`, conditional expressions, `for`, `range`, `while` | Apply decisions and repeat work over data |
| 5–6 · Functions and lambdas | Parameters, docstrings, `return`, printing and one-expression anonymous functions | Reuse logic and distinguish a returned value from a displayed message |
| 7 · Built-ins | `len`, `min`, `max`, `sorted`, `map`, `filter` | Express common transformations clearly |
| 8 · Practical iteration | List comprehensions, `enumerate`, `zip` | Filter compactly, attach indices and pair related sequences |

## 💻 Important code blocks

### Character counting with a dictionary — cells 128 and 138

```python
stri = "Machine learning"
char_count = {}
for c in stri:
    char_count[c] = char_count.get(c, 0) + 1
print(char_count)
```

Each character becomes a key; `get(c, 0)` supplies the initial count. The second implementation uses `setdefault(c, 0)` before incrementing and produces the same counts. This bridges loops, strings, membership and mutable state.

### Functions applied across collections — cells 215–220

```python
def cubic(number):
    return number ** 3

num_list = [0, 1, 2, 3, 4]
list(map(cubic, num_list))
```

The actual result is `[0, 1, 8, 27, 64]`. `filter(odd_check, num_list)` in the separate odd-number example keeps `[1, 5, 7, 9, 11]`; the predicate selects elements, whereas `map` transforms every element.

### Comprehension versus a loop — cells 225–231

```python
nums = range(1, 20)
even_nums = [num for num in nums if num % 2 == 0]
print(even_nums)
```

The output is `[2, 4, 6, 8, 10, 12, 14, 16, 18]`, matching the longer loop. Filtering day names produces `['Tuesday', 'Thursday']`.

### Pairing related data — cells 241–243

```python
name = ['Jessy', 'Joe', 'Jeannette']
role = ['ML Engineer', 'Web Developer', 'Data Engineer']
zipped_name_role = zip(name, role)
list(zipped_name_role)
```

The output pairs each person with the corresponding role. `enumerate(seasons, start=1)` similarly pairs each season with its display number.

## 🔎 Reading the results correctly

Assignment cells usually have no output; expression cells show their last value. A `zip` object representation is a real iterator, not missing data. Converting it to a list materializes its pairs. Tuple mutation examples are deliberately commented out to explain errors; they are not failing executed operations. All executable examples were rerun in sequence.

## ⚠️ Pitfalls & corrections to keep in mind

- Slices exclude the stop index; `range(1, 20)` therefore stops at 19.
- `list.sort()` mutates the list; `sorted()` returns a new sorted list. `pop` removes by position; `remove` removes by value.
- Lists and dictionaries are mutable. Strings and tuples are immutable; sets remove duplicates and do not provide list-style positions.
- Python supports many types beyond the three introductory examples, including Boolean, collection, complex and user-defined types.
- Triple-quoted text is a string literal, and a function's leading string is its docstring; it is not literally a multi-line comment mechanism.
- `map`, `filter`, `zip` and `enumerate` are iterators; consuming one can change what a later read returns. `zip` stops at the shorter input by default.
- A `while` loop needs a condition-changing update. A function that only prints returns `None` implicitly.

## ✅ Run notes

Both blocking `input(str)` prompts were replaced by the explicit, printed sample **Machine learning** so a fresh unattended run finishes predictably. The counting logic is unchanged. The notebook uses the Python standard library only. No model accuracy or generalization claim applies here.

## 🧠 Quick self-check

Explain why the comprehension stops at 18, rewrite the character counter with `setdefault`, and describe what happens if `list(zipped_name_role)` is evaluated a second time.
