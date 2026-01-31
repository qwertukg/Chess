# Secret Prelude Chess

Веб‑шахматы с «секретной прелюдией» из N ходов для каждого игрока. Все ходы — только мышью.

## Запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Откройте `http://localhost:8000`.

### Stockfish (опционально)
Укажите путь к движку через `STOCKFISH_PATH`:

```bash
export STOCKFISH_PATH=/path/to/stockfish
```

Если переменная не задана, бот ходит случайно допустимыми ходами.

## Правила прелюдии (сервер — авторитет)

- Игроки формируют списки `W[0..N-1]` и `B[0..N-1]`.
- Сервер применяет ходы из стартовой позиции:
  1) Пробует `W[i]`. Если ход нелегален — белые прекращают.
  2) Пробует `B[i]`. Если ход нелегален — чёрные прекращают.
  3) Если один игрок «сломался», второй продолжает оставшиеся ходы, пока они легальны.
- Итоговая позиция становится стартом открытой игры.
- В раскрытии показываются применённые ходы и причина остановки (если была).

## WebSocket протокол

Endpoint: `/ws/{roomId}?playerId=...`

Сообщения клиент → сервер:

```json
{ "type": "prelude_move_add", "uci": "e2e4" }
{ "type": "prelude_move_undo" }
{ "type": "prelude_move_clear" }
{ "type": "prelude_ready" }
{ "type": "move", "uci": "e2e4" }
{ "type": "resign" }
```

Сообщения сервер → клиент:

```json
{ "type": "join", "playerId": "...", "color": "white", "state": { ... } }
{ "type": "state", "state": { ... } }
{ "type": "prelude_plan", "moves": ["e2e4", "g1f3"] }
{ "type": "prelude_reveal", "result": { ... } }
{ "type": "move", "uci": "e2e4", "san": "e4", "fen": "..." }
{ "type": "illegal_move", "reason": "..." }
{ "type": "game_over", "reason": "resign", "winner": "white" }
```

### `state` (публичное состояние комнаты)

```json
{
  "roomId": "room_...",
  "mode": "pvp",
  "n": 4,
  "state": "prelude",
  "createdAt": 1738327800.0,
  "players": {
    "player_...": { "color": "white" }
  },
  "fen": "...",
  "prelude": { "readyWhite": false, "readyBlack": false }
}
```

## Тесты

```bash
pytest
```

## Примечания

- Валидация прелюдии на сервере делается через «превью» доску игрока (только его ходы).
- Во время прелюдии половина соперника скрыта и неактивна на клиенте.
- Состояние комнат хранится в памяти (MVP).
