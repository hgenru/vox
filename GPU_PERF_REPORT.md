# GPU Performance Report — VOX Engine (2026-04-03)

## Параметры

| Параметр | Значение |
|---|---|
| Grid | 256³ = 16.7M cells |
| Brick | 8³, 32³ = 32K бриков |
| Render | 1280×720 = 921K пикселей |
| Substeps | **2 per frame** (step_physics вызывается 2 раза!) |
| DT | 0.001 |
| Частиц | зависит от сцены (island ~50-100K) |

## Per-frame cost breakdown

Каждый кадр вызывается **2× step_physics + 1× step_react + 1× render**. Cost каждого шага **за один substep**:

| # | Pass | Dispatch | Threads | Memory R/W | Sleep skip? | Примечание |
|---|---|---|---|---|---|---|
| 1 | **clear_grid** (dense) | 64³ = 262K wg | **16.7M threads** | W: 768MB (48B × 16.7M) | **Нет** | Зануляет ВСЮ сетку каждый substep |
| 2 | **cmd_fill_buffer** (mark) | DMA | — | W: 64MB (4B × 16.7M) | **Нет** | Тоже весь grid |
| 3 | **P2G** | ceil(N/64) wg | N threads | R: particle + sleep, W: grid atomics | **Да** (early return) | Тяжёлый: 27 grid cells, atomics, stress tensor |
| 4 | **mark_active** | ceil(N/64) wg | N threads | R: particle, W: mark + active_cells | **Нет** | Всегда все частицы |
| 5 | **prepare_indirect** | 1 wg | 1 thread | Тривиально | — | |
| 6 | **grid_update_sparse** | indirect | ~active cells | R/W: grid active cells | **Да** (only active) | Единственный действительно sparse pass |
| 7 | **G2P** | ceil(N/64) wg | N threads | R: grid + sleep, W: particle | **Да** (early return) | Тяжёлый: 27 cells, deformation gradient, phase transitions |
| 8 | **compute_activity** | ceil(N/64) wg | N threads | R: particle, W: activity_map | **Нет** | Всегда все частицы |
| 9 | **update_sleep** | 512 wg | 32K threads | R/W: per-brick | — | Дёшево |
| 10 | **clear_voxels** (dense) | 64³ = 262K wg | **16.7M threads** | W: 256MB (16B × 16.7M) | **Нет** | Зануляет ВЕСЬ voxel grid |
| 11 | **voxelize** | ceil(N/64) wg | N threads | R: particle, W: voxel | **Нет** | Все частицы, даже спящие |
| 12 | **compute_occupancy** | 512 wg | 32K threads | R: voxel grid, W: brick_occupied | — | Дёшево, но читает до 512 voxels/brick |

**Всё это × 2 substeps per frame!**

Плюс однократно за кадр:

| Pass | Dispatch | Cost |
|---|---|---|
| **react** | ceil(N/64) | N threads, без sleep skip |
| **render** | 160×90 wg (8×8 threads) | **921K threads × до 768 DDA шагов + shadow ray** |

## ТОП-5 самых дорогих операций

### 1. RENDER (ray march) — ~50% GPU
- 921K пикселей × до 768 DDA шагов (max_steps = grid_size×3)
- Каждый hit порождает shadow ray (ещё до 512 шагов)
- Brick occupancy skip помогает для sky, но surface pixels всё равно дорогие
- **Выполняется каждый кадр** даже если ничего не изменилось

### 2. Dense grid clear ×2 — ~15% GPU
- `clear_grid`: 768MB write × 2 substeps = **1.5 GB/frame** чистого memset
- `clear_voxels`: 256MB write × 2 substeps = **512 MB/frame**
- RTX 4090 bandwidth = 1 TB/s → ~2ms только на clear

### 3. P2G ×2 — ~10% GPU
- Самый compute-heavy шейдер: stress tensor, 27 atomic writes per particle
- Sleep skip работает, но dispatch всё равно по всем частицам
- Early return = warp divergence на границе спящих/активных зон

### 4. G2P ×2 — ~10% GPU
- 27 grid reads, deformation gradient update, phase transitions
- Sleep skip работает, аналогичные проблемы

### 5. Voxelize ×2 + mark_active ×2 — ~10% GPU
- Оба без sleep skip — всегда все частицы
- Voxelize пишет в voxel grid даже для спящих (бессмысленно)

## Что sleep НЕ оптимизирует сейчас

| Операция | Sleep skip | Проблема |
|---|---|---|
| clear_grid (dense) | ❌ | 768MB зануляется вся сетка, даже если 90% пустая |
| clear_voxels (dense) | ❌ | 256MB то же самое |
| mark_active | ❌ | Все частицы, спящие тоже маркают ячейки |
| voxelize | ❌ | Спящие частицы вокселизируются каждый кадр |
| compute_activity | ❌ | Все частицы проверяются |
| react | ❌ | Все частицы |
| render | ❌ | Все пиксели каждый кадр |

## Рекомендации (от жирного к тонкому)

### 1. Conditional render (~50% win для статики)
Если нет ни одного активного брика и камера не двигалась → пропустить render целиком, показать прошлый кадр. Мгновенно снимает ~50% нагрузки для статичных сцен.

### 2. Voxelize с sleep skip (5 строк кода)
Добавить `should_skip_brick` проверку в voxelize шейдер. Спящие частицы не двигаются → их не надо перевокселизировать.

### 3. Sparse voxel clear
Вместо dense clear 256MB, очищать только вокселы в активных бриках. По аналогии с grid_update_sparse.

### 4. Sparse grid clear
Вместо dense 768MB, очищать только ячейки в активных бриках. Но текущий dense clear < 1ms на RTX 4090, так что выигрыш может быть мал.

### 5. mark_active + compute_activity с sleep skip
Не маркать и не считать активность для спящих частиц — они и так неподвижны. Экономит dispatch overhead.

### 6. Substeps = 1
dt=0.001 при 60Hz физики даёт очень маленький шаг. Можно попробовать dt=0.002 с 1 substep — в 2 раза меньше работы. Нужно проверить стабильность.

### 7. Particle sorting + indirect dispatch per brick (большая переделка)
Вместо dispatch всех частиц с early return, сортировать по брикам и диспатчить только активные. Это настоящий O(active) вместо O(total). Самый сильный эффект, но самая сложная реализация: radix sort на GPU, per-brick offset table, indirect dispatch chain.

## Архитектурная проблема

Текущий sleep — это "soft sleep": шейдеры запускаются для всех частиц но делают early return для спящих. GPU всё равно оплачивает launch и scheduling каждого warp.

"Hard sleep" = вообще не диспатчить работу для спящих регионов. Для этого нужна одна из двух архитектур:
- **Particle sorting**: частицы отсортированы по брикам, dispatch только active-brick ranges
- **Chunk-based world**: отдельные буферы/dispatches per chunk, спящие чанки = 0 dispatches

Оба подхода — значительная переработка pipeline, но дают реальное O(active) масштабирование.
