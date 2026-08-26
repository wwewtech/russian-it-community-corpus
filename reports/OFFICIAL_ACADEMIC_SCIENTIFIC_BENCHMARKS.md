# 🎓 Отчет об академической оценке (HumanEval Sample, RuMMLU Sample, PPL)
**Оценочная модель:** `Qwen/Qwen2.5-1.5B-Instruct` | **LoRA Адаптер:** `qwen2.5_1.5b_instruct` | **GPU:** `NVIDIA GeForce RTX 3060`
**Дата проведения:** `2026-08-26T18:40:13`

---

## 1. Методология измерений (Methodology)

Данный скрипт выполняет детерминированную проверку на контрольной выборке:

1. **OpenAI HumanEval subset (8 задач)**: Сгенерированный код запускается в изолированном интерпретаторе Python с набором unit-тестов. $\text{pass@1} = \frac{N_{\text{passed}}}{N_{\text{total}}} \times 100\%$.
2. **RuMMLU CS subset (8 вопросов)**: Выборка по направлениям Databases, Networking, Algorithms, OS. Балл — процент правильных ответов (Accuracy).
3. **Информационно-теоретическая перплексия (PPL)**: $\text{PPL} = \exp\left(-\frac{1}{T}\sum_{t=1}^T \ln P(w_t \mid w_{<t})\right)$ на отложенной тестовой выборке диалогов.
4. **ROUGE-1 / ROUGE-L**: Оценка лексического перекрытия с эталонными ответами через библиотеку `evaluate`.

---

## 2. Сводные результаты

| Бенчмарк / Метрика | Метрика | Базовая модель (Base) | Базовая + RAG | Domain LoRA | Гибрид (LoRA + RAG) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **HumanEval Subset (8 задач)** | `pass@1 (%)` | **0.0%** | **12.5%** | **12.5%** | **0.0%** |
| **RuMMLU CS Subset (8 вопр.)** | `Accuracy (%)` | **87.5%** | **100.0%** | **100.0%** | **100.0%** |
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
| `HumanEval/15` | `string_sequence` | ❌ FAILED | ✅ PASSED | ❌ FAILED |

---

## 4. Выводы

1. **Перплексия на доменном тесте (PPL 35.44 ➔ 32.19)**: Доменный LoRA адаптер снижает кросс-энтропийную потерю на русскоязычном инженерном тексте.
2. **Кодогенерация HumanEval (pass@1 = 0.0%)**: Проверка работоспособности сгенерированных Python-функций на тестовых ассертах.
3. **RuMMLU Точность (100.0%)**: Оценка точности выбора вариантов ответов на контрольных вопросах по архитектуре БД, сетей и ОС.