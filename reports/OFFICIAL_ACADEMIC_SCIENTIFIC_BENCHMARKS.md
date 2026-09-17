# 🎓 Отчет об академической оценке (HumanEval Sample, RuMMLU Sample, PPL)
**Оценочная модель:** `Qwen/Qwen2.5-1.5B-Instruct` | **LoRA Адаптер:** `qwen2.5_1.5b_instruct` | **GPU:** `NVIDIA GeForce RTX 3060`
**Дата проведения:** `2026-09-17T17:16:34`

---

## 1. Методология измерений (Methodology)

Данный скрипт выполняет детерминированную проверку на контрольной выборке:

1. **OpenAI HumanEval subset (40 задач)**: Сгенерированный код запускается в изолированном интерпретаторе Python с набором unit-тестов. $\text{pass@1} = \frac{N_{\text{passed}}}{N_{\text{total}}} \times 100\%$.
2. **RuMMLU CS subset (50 вопросов)**: Выборка по направлениям Databases, Networking, Algorithms, OS, Distributed Systems, Security, Programming Languages. Балл — процент правильных ответов (Accuracy).
   Выборки расширены с 8 до 40 задач / 50 вопросов для статистической силы: при N=8 доверительный интервал Вильсона (95%) достигает ±20 п.п., при N=50 — уже ±7.8 п.п. Все публикуемые точности сопровождаются интервалами.
3. **Информационно-теоретическая перплексия (PPL)**: $\text{PPL} = \exp\left(-\frac{1}{T}\sum_{t=1}^T \ln P(w_t \mid w_{<t})\right)$ на отложенной тестовой выборке диалогов.
4. **ROUGE-1 / ROUGE-L**: Оценка лексического перекрытия с эталонными ответами через библиотеку `evaluate`.

---

## 2. Сводные результаты

| Бенчмарк / Метрика | Метрика | Базовая модель (Base) | Базовая + RAG | Domain LoRA | Гибрид (LoRA + RAG) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **HumanEval Subset (40 задач)** | `pass@1 (%)` | **7.5%** | **2.5%** | **5.0%** | **15.0%** |
| **RuMMLU CS Subset (50 вопр.)** | `Accuracy (%)` | **88.0%** | **90.0%** | **88.0%** | **92.0%** |
| **HumanEval pass@1 — 95% Wilson CI** | `интервал` | 2.6–19.9% | 0.4–12.9% | 1.4–16.5% | 7.1–29.1% |
| **RuMMLU accuracy — 95% Wilson CI** | `интервал` | 76.2–94.4% | 78.6–95.7% | 76.2–94.4% | 81.2–96.8% |
| **Test Set Perplexity** | `PPL (ниже = лучше)` | `35.44` | N/A | **`32.19`** | **`32.19`** |
| **ROUGE-1 F1** | `Overlap (%)` | `42.6%` | N/A | **`45.4%`** | **`45.4%`** |
| **ROUGE-L F1** | `LCS Overlap (%)` | `36.4%` | N/A | **`38.8%`** | **`38.8%`** |

---

## 3. Детальный разбор выполнения HumanEval subset

| Задача HumanEval | Сигнатура функции | Unit-тесты Base | Unit-тесты LoRA | Unit-тесты Hybrid |
| :--- | :--- | :---: | :---: | :---: |
| `HumanEval/0` | `has_close_elements` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/2` | `truncate_number` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/3` | `below_zero` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/4` | `mean_absolute_deviation` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/5` | `intersperse` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/8` | `sum_product` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/11` | `string_xor` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/15` | `string_sequence` | ❌ FAILED | ✅ PASSED | ✅ PASSED |
| `HumanEval/1` | `separate_paren_groups` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/6` | `parse_nested_parens` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/7` | `filter_by_substring` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/9` | `rolling_max` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/10` | `make_palindrome` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/12` | `longest` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/13` | `greatest_common_divisor` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/14` | `all_prefixes` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/16` | `count_distinct_characters` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/17` | `parse_music` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/18` | `how_many_times` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/19` | `sort_numbers` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/20` | `find_closest_elements` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/21` | `rescale_to_unit` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/22` | `filter_integers` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/23` | `strlen` | ❌ FAILED | ❌ FAILED | ✅ PASSED |
| `HumanEval/24` | `largest_divisor` | ✅ PASSED | ❌ FAILED | ✅ PASSED |
| `HumanEval/25` | `factorize` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/26` | `remove_duplicates` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/27` | `flip_case` | ❌ FAILED | ❌ FAILED | ✅ PASSED |
| `HumanEval/28` | `concatenate` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/29` | `filter_by_prefix` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/30` | `get_positive` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/31` | `is_palindrome` | ✅ PASSED | ❌ FAILED | ✅ PASSED |
| `HumanEval/33` | `sort_third` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/34` | `unique` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/35` | `max_element` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/36` | `fizz_buzz` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/37` | `sort_even` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/39` | `prime_fib` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/40` | `triples_sum_to_zero` | ❌ FAILED | ❌ FAILED | ❌ FAILED |
| `HumanEval/41` | `car_race_collision` | ✅ PASSED | ✅ PASSED | ✅ PASSED |

---

## 4. Выводы

1. **Перплексия на доменном тесте (PPL 35.44 ➔ 32.19)**: Доменный LoRA адаптер снижает кросс-энтропийную потерю на русскоязычном инженерном тексте.
2. **Кодогенерация HumanEval (pass@1 = 15.0%)**: Проверка работоспособности сгенерированных Python-функций на тестовых ассертах.
3. **RuMMLU Точность (92.0%)**: Оценка точности выбора вариантов ответов на контрольных вопросах по архитектуре БД, сетей и ОС.
