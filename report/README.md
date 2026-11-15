# Отчёт

Исходник отчёта: `REPORT.tex`. Готовый `REPORT.pdf` лежит в корне репозитория.

## Сборка

Нужен [tectonic](https://tectonic-typesetting.github.io/) (сам скачивает пакеты LaTeX при первом запуске) и шрифты PT Serif, PT Sans, PT Mono.

Из корня репозитория (или `make -C report`):

```bash
tectonic -X compile report/REPORT.tex --outdir .
```

## Откуда числа

Все таблицы взяты из файлов оценщика в `artifacts/`: `eval_*.json`, `index_stats.json`, `extraction_audit.json`. После нового прогона (`make eval-all`, `make eval-extraction`) числа в `REPORT.tex` нужно обновить вручную.
