#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AlphaZero-lite для шахмат:
- Policy+Value сеть (PyTorch)
- MCTS с батчингом оценок (ускорение, особенно на MPS)
- Self-play, replay buffer
- Data augmentation: зеркалирование по вертикали (A<->H) (безопасно для шахмат)
- Оценка относительного Elo против базового GreedyBot

Требования:
  pip install torch python-chess
"""

import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Device (M2 Pro: MPS)
# =========================
def pick_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# =========================
# Конфиг
# =========================
@dataclass
class Config:
    seed: int = 1
    device: str = pick_device()

    # action space: (from,to,promo) where promo in {None,Q,R,B,N} => 5
    action_size: int = 64 * 64 * 5

    # MCTS
    mcts_sims: int = 160
    cpuct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    temperature_moves: int = 20
    mcts_eval_batch: int = 64  # сколько листьев прогоняем сетью за один батч

    # Self-play / training loop
    games_per_iter: int = 12
    max_game_plies: int = 300
    # Outcome shaping / efficiency
    draw_value_scale: float = 0.2  # масштаб "материального" сигнала для ничьих/усечений
    material_value_scale: float = 6.0  # чем больше, тем слабее влияние материала
    resign_enabled: bool = True
    resign_material_threshold: float = 8.0  # разница материала (в пешках) для досрочного "resign"
    resign_min_plies: int = 40  # не сдаваться слишком рано

    # Replay / training
    replay_max: int = 200000
    train_steps_per_iter: int = 400
    batch_size: int = 256
    epochs_per_iter: int = 1  # мы делаем train_steps_per_iter, epochs тут условны
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    log_train_every: int = 50

    # Model
    channels: int = 128
    res_blocks: int = 8

    # Eval vs baseline
    eval_games: int = 10
    eval_every_iters: int = 2  # оценивать не каждую итерацию
    baseline_name: str = "GreedyBot"

    # Checkpoints
    out_dir: str = "out_azlite"
    save_every_iters: int = 1


CFG = Config()

random.seed(CFG.seed)
torch.manual_seed(CFG.seed)


# =========================
# Кодирование доски
# =========================
PLANES = 18  # 12 pieces + stm + castling4 + ep
PIECE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

PROMO_MAP = [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
PROMO_IDX = {None: 0, chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 3, chess.KNIGHT: 4}


def encode_board(board: chess.Board) -> torch.Tensor:
    """
    (18, 8, 8) float32
    0..5: белые фигуры
    6..11: чёрные фигуры
    12: side-to-move (1 если белые ходят)
    13..16: castling rights (WK,WQ,BK,BQ)
    17: en-passant target square
    """
    x = torch.zeros((PLANES, 8, 8), dtype=torch.float32)

    for sq, piece in board.piece_map().items():
        r = 7 - chess.square_rank(sq)
        c = chess.square_file(sq)
        base = PIECE_TO_PLANE[piece.piece_type]
        plane = base if piece.color == chess.WHITE else base + 6
        x[plane, r, c] = 1.0

    x[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    x[13, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    x[14, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    x[15, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    x[16, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    if board.ep_square is not None:
        r = 7 - chess.square_rank(board.ep_square)
        c = chess.square_file(board.ep_square)
        x[17, r, c] = 1.0

    return x


def move_to_action(move: chess.Move) -> int:
    f = move.from_square
    t = move.to_square
    p = PROMO_IDX.get(move.promotion, 0)
    return (f * 64 + t) * 5 + p


def action_to_move(action: int) -> chess.Move:
    ft, p = divmod(action, 5)
    f, t = divmod(ft, 64)
    promo = PROMO_MAP[p]
    return chess.Move(f, t, promotion=promo)


def legal_actions(board: chess.Board) -> List[int]:
    return [move_to_action(mv) for mv in board.legal_moves]


# =========================
# Зеркалирование A<->H (data augmentation)
# =========================
# В шахматах симметрии "как у го" НЕ работают полностью из-за направления пешек.
# Безопасная симметрия: отражение по вертикали (файлы A<->H), оно не меняет направление пешек.
# Нужно:
# - отразить квадраты (file -> 7-file)
# - отразить en-passant квадрат
# - поменять местами kingside/queenside castling flags (они завязаны на сторону доски)
# - отразить policy (действия from/to + промо остаётся тем же)
def mirror_square(sq: int) -> int:
    f = chess.square_file(sq)
    r = chess.square_rank(sq)
    mf = 7 - f
    return chess.square(mf, r)


def mirror_move(mv: chess.Move) -> chess.Move:
    return chess.Move(
        mirror_square(mv.from_square),
        mirror_square(mv.to_square),
        promotion=mv.promotion,
    )


def mirror_action(action: int) -> int:
    mv = action_to_move(action)
    mm = mirror_move(mv)
    return move_to_action(mm)


def mirror_board(board: chess.Board) -> chess.Board:
    b = chess.Board(fen=board.fen())

    # python-chess не даёт "простого" мутационного API для зеркала через piece_map с сохранением всего,
    # поэтому пересоберём доску руками.
    nb = chess.Board.empty()
    nb.clear_stack()

    nb.turn = b.turn

    # поставить фигуры, отражая квадраты
    for sq, piece in b.piece_map().items():
        nb.set_piece_at(mirror_square(sq), piece)

    # castling rights: при зеркале K<->Q
    w_k = b.has_kingside_castling_rights(chess.WHITE)
    w_q = b.has_queenside_castling_rights(chess.WHITE)
    b_k = b.has_kingside_castling_rights(chess.BLACK)
    b_q = b.has_queenside_castling_rights(chess.BLACK)

    rights = 0
    # после зеркала kingside и queenside меняются местами
    if w_q:
        rights |= chess.BB_H1  # маркер не используется напрямую; проще через set_castling_fen
    # python-chess ожидает castling fen, сделаем строкой:
    cast = ""
    cast += "K" if w_q else ""
    cast += "Q" if w_k else ""
    cast += "k" if b_q else ""
    cast += "q" if b_k else ""
    nb.set_castling_fen(cast if cast else "-")

    # en-passant
    nb.ep_square = mirror_square(b.ep_square) if b.ep_square is not None else None

    # halfmove/fullmove counters
    nb.halfmove_clock = b.halfmove_clock
    nb.fullmove_number = b.fullmove_number

    # sanity: если вдруг позиция некорректная, пусть упадёт сразу
    _ = nb.fen()
    return nb


def augment_sample(state_planes: torch.Tensor, policy: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Возвращает список (state, policy) для обучения:
    - исходный
    - зеркальный по вертикали
    """
    out = [(state_planes, policy)]
    # state: (18,8,8) - зеркалим по оси x (file), то есть flip по ширине (dim=2)
    s_m = torch.flip(state_planes, dims=[2]).clone()
    # castling rights: при зеркале K<->Q для обеих сторон
    s_m[[13, 14]] = s_m[[14, 13]]
    s_m[[15, 16]] = s_m[[16, 15]]

    # policy: отражаем action indices
    pi_m = torch.zeros_like(policy)
    nz = torch.nonzero(policy, as_tuple=False).flatten()
    for idx in nz.tolist():
        pi_m[mirror_action(idx)] = policy[idx]
    out.append((s_m, pi_m))
    return out


# =========================
# Policy+Value сеть
# =========================
class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        r = x
        x = F.relu(self.b1(self.c1(x)))
        x = self.b2(self.c2(x))
        return F.relu(x + r)


class PolicyValueNet(nn.Module):
    def __init__(self, channels=96, blocks=6, action_size=CFG.action_size):
        super().__init__()
        self.action_size = action_size
        self.stem = nn.Sequential(
            nn.Conv2d(PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.res = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

        # policy
        self.p_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.p_fc = nn.Linear(32 * 8 * 8, action_size)

        # value
        self.v_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.v_fc1 = nn.Linear(32 * 8 * 8, 128)
        self.v_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.res(self.stem(x))
        p = self.p_head(x).flatten(1)
        logits = self.p_fc(p)
        v = self.v_head(x).flatten(1)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v)).squeeze(1)
        return logits, v


# =========================
# Replay Buffer
# =========================
class ReplayBuffer:
    def __init__(self, max_size: int):
        self.max = max_size
        self.buf: List[Tuple[torch.Tensor, torch.Tensor, float]] = []
        self.pos = 0

    def __len__(self):
        return len(self.buf)

    def add(self, state: torch.Tensor, policy: torch.Tensor, value: float):
        item = (state.cpu(), policy.cpu(), float(value))
        if len(self.buf) < self.max:
            self.buf.append(item)
        else:
            self.buf[self.pos] = item
            self.pos = (self.pos + 1) % self.max

    def sample(self, n: int) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
        n = min(n, len(self.buf))
        return random.sample(self.buf, n)


# =========================
# Базовый бот для оценки (GreedyBot)
# =========================
PIECE_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def material_balance(board: chess.Board) -> float:
    """Материальный баланс: белые - чёрные, в пешечных единицах."""
    diff = 0.0
    for p, v in PIECE_VALUE.items():
        if v == 0:
            continue
        diff += len(board.pieces(p, chess.WHITE)) * v
        diff -= len(board.pieces(p, chess.BLACK)) * v
    return diff


def drawish_value(board: chess.Board) -> float:
    """Небольшой "материальный" сигнал для ничьих/усечений, чтобы обучать value."""
    if CFG.draw_value_scale <= 0 or CFG.material_value_scale <= 0:
        return 0.0
    diff = material_balance(board)
    return float(CFG.draw_value_scale * math.tanh(diff / CFG.material_value_scale))


def shaped_game_value_white(board: chess.Board, truncated: bool) -> float:
    """
    Возвращает value с точки зрения белых.
    Для ничьих/усечений даём слабый материал-сигнал.
    """
    res = board.result(claim_draw=True)
    if res == "1-0":
        return 1.0
    if res == "0-1":
        return -1.0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    if truncated or res == "1/2-1/2":
        return drawish_value(board)
    return 0.0


def resign_result_from_material(board: chess.Board) -> Optional[float]:
    if not CFG.resign_enabled:
        return None
    diff = material_balance(board)
    if diff >= CFG.resign_material_threshold:
        return 1.0
    if diff <= -CFG.resign_material_threshold:
        return -1.0
    return None


def greedy_bot_move(board: chess.Board) -> chess.Move:
    moves = list(board.legal_moves)
    if not moves:
        return chess.Move.null()

    best = []
    best_gain = -999

    for mv in moves:
        if board.is_capture(mv):
            captured = board.piece_at(mv.to_square)
            gain = PIECE_VALUE.get(captured.piece_type, 0) if captured else 0
            # чуть бонус за шах
            board.push(mv)
            gives_check = board.is_check()
            board.pop()
            if gives_check:
                gain += 0.25
            if gain > best_gain:
                best_gain = gain
                best = [mv]
            elif gain == best_gain:
                best.append(mv)

    if best_gain > 0 and best:
        return random.choice(best)

    # иначе случайный, но с предпочтением развивающих ходов (очень грубо)
    return random.choice(moves)


# =========================
# MCTS с батчингом оценок
# =========================
class Node:
    __slots__ = ("prior", "to_play", "N", "W", "children")

    def __init__(self, prior: float, to_play: bool):
        self.prior = float(prior)
        self.to_play = bool(to_play)
        self.N = 0
        self.W = 0.0
        self.children: Optional[Dict[int, "Node"]] = None

    def Q(self) -> float:
        return 0.0 if self.N == 0 else self.W / self.N


def softmax_masked(logits_1d: torch.Tensor, legal: List[int]) -> torch.Tensor:
    mask = torch.full_like(logits_1d, -1e9)
    idx = torch.tensor(legal, device=logits_1d.device, dtype=torch.long)
    mask[idx] = 0.0
    x = logits_1d + mask
    return F.softmax(x, dim=0)


@torch.no_grad()
def expand_with_policy_value(
    net: PolicyValueNet,
    boards: List[chess.Board],
    nodes: List[Node],
    add_dirichlet_to_root: bool,
) -> List[float]:
    """
    Батчево оценивает boards, заполняет node.children, возвращает values (с точки зрения side-to-move каждого board).
    add_dirichlet_to_root применяется только к первой паре (boards[0], nodes[0]) если True.
    """
    x = torch.stack([encode_board(b) for b in boards]).to(CFG.device)
    logits, values = net(x)  # logits: (B, A), values: (B,)
    out_vals: List[float] = []

    for i, (b, node) in enumerate(zip(boards, nodes)):
        legal = legal_actions(b)
        probs = softmax_masked(logits[i], legal)

        if add_dirichlet_to_root and i == 0 and len(legal) > 0:
            alpha = torch.full((len(legal),), CFG.dirichlet_alpha, device=CFG.device)
            try:
                noise = torch.distributions.Dirichlet(alpha).sample()
            except NotImplementedError:
                # MPS doesn't implement Dirichlet sampling; fall back to CPU for noise only.
                noise = torch.distributions.Dirichlet(alpha.cpu()).sample().to(CFG.device)
            probs_legal = probs[torch.tensor(legal, device=CFG.device)]
            probs_legal = (1 - CFG.dirichlet_eps) * probs_legal + CFG.dirichlet_eps * noise
            probs = probs.clone()
            probs[torch.tensor(legal, device=CFG.device)] = probs_legal

        node.children = {}
        for a in legal:
            node.children[a] = Node(prior=float(probs[a].item()), to_play=not b.turn)

        out_vals.append(float(values[i].item()))

    return out_vals


def ucb_score(parent: Node, child: Node) -> float:
    pb_c = CFG.cpuct * child.prior * math.sqrt(parent.N + 1e-8) / (1 + child.N)
    return child.Q() + pb_c


def select_action(node: Node) -> int:
    assert node.children is not None
    best_a, best_s = -1, -1e18
    for a, ch in node.children.items():
        s = ucb_score(node, ch)
        if s > best_s:
            best_s = s
            best_a = a
    return best_a


def terminal_value_for_side_to_move(board: chess.Board) -> float:
    res = board.result(claim_draw=True)
    if res == "1-0":
        v_white = 1.0
    elif res == "0-1":
        v_white = -1.0
    else:
        v_white = 0.0
    # вернуть "с точки зрения side-to-move"
    return v_white if board.turn == chess.WHITE else -v_white


def backprop(path: List[Node], leaf_value: float):
    v = leaf_value
    for n in reversed(path):
        n.N += 1
        n.W += v
        v = -v


@torch.no_grad()
def run_mcts_batched(net: PolicyValueNet, root_board: chess.Board) -> Tuple[Node, Dict[int, int]]:
    """
    MCTS, но оценка листьев батчится.
    """
    # 1) создаём root и сразу расширяем (с Dirichlet noise)
    root = Node(prior=1.0, to_play=root_board.turn)
    root.children = {}
    _ = expand_with_policy_value(net, [root_board], [root], add_dirichlet_to_root=True)

    pending_paths: List[List[Node]] = []
    pending_boards: List[chess.Board] = []
    pending_nodes: List[Node] = []

    def flush_pending():
        if not pending_boards:
            return
        vals = expand_with_policy_value(net, pending_boards, pending_nodes, add_dirichlet_to_root=False)
        for path, v in zip(pending_paths, vals):
            backprop(path, v)
        pending_paths.clear()
        pending_boards.clear()
        pending_nodes.clear()

    # 2) симуляции
    for _ in range(CFG.mcts_sims):
        b = root_board.copy(stack=False)
        node = root
        path = [node]

        # selection until leaf or terminal
        while node.children is not None and len(node.children) > 0:
            a = select_action(node)
            mv = action_to_move(a)
            if mv not in b.legal_moves:
                break
            b.push(mv)
            node = node.children[a]
            path.append(node)
            if b.is_game_over():
                break

        # evaluate / expand
        if b.is_game_over():
            v = terminal_value_for_side_to_move(b)
            backprop(path, v)
        else:
            # leaf node: если не расширен — добавим в pending
            if node.children is None:
                node.children = {}
            pending_paths.append(path)
            pending_boards.append(b.copy(stack=False))
            pending_nodes.append(node)

            if len(pending_boards) >= CFG.mcts_eval_batch:
                flush_pending()

    flush_pending()

    counts = {a: ch.N for a, ch in (root.children or {}).items()}
    return root, counts


def counts_to_policy(counts: Dict[int, int]) -> torch.Tensor:
    pi = torch.zeros((CFG.action_size,), dtype=torch.float32)
    s = sum(counts.values())
    if s <= 0:
        return pi
    for a, c in counts.items():
        pi[a] = c / s
    return pi


def sample_from_counts(counts: Dict[int, int], temperature: float) -> int:
    items = list(counts.items())
    actions = [a for a, _ in items]
    n = torch.tensor([c for _, c in items], dtype=torch.float32)
    if temperature <= 1e-6:
        return actions[int(torch.argmax(n).item())]
    n = n ** (1.0 / temperature)
    p = n / (n.sum() + 1e-9)
    idx = torch.multinomial(p, 1).item()
    return actions[idx]


# =========================
# Self-play
# =========================
def game_result_white(board: chess.Board) -> float:
    res = board.result(claim_draw=True)
    if res == "1-0":
        return 1.0
    if res == "0-1":
        return -1.0
    return 0.0


@torch.no_grad()
def self_play_game(net: PolicyValueNet) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor, float]], Dict[str, float]]:
    """
    Возвращает список обучающих примеров (state_planes, policy_target, value_target)
    value_target — с точки зрения side-to-move в этом состоянии.
    """
    board = chess.Board()
    traj: List[Tuple[torch.Tensor, torch.Tensor, bool]] = []
    stats = {
        "plies": 0,
        "legal_sum": 0,
        "entropy_sum": 0.0,
        "top1_sum": 0.0,
        "root_q_sum": 0.0,
        "samples_n": 0,
        "z_abs_sum": 0.0,
        "truncated": 0.0,
        "resigned": 0.0,
        "result": 0.0,
    }

    ply = 0
    resigned_result: Optional[float] = None
    while not board.is_game_over() and ply < CFG.max_game_plies:
        if ply >= CFG.resign_min_plies:
            rr = resign_result_from_material(board)
            if rr is not None:
                resigned_result = rr
                stats["resigned"] = 1.0
                break

        root, counts = run_mcts_batched(net, board)

        temp = 1.0 if ply < CFG.temperature_moves else 1e-6
        action = sample_from_counts(counts, temperature=temp)
        mv = action_to_move(action)

        # fallback
        if mv not in board.legal_moves:
            leg = legal_actions(board)
            if not leg:
                break
            action = max(((a, counts.get(a, 0)) for a in leg), key=lambda x: x[1])[0]
            mv = action_to_move(action)

        s = encode_board(board)
        pi = counts_to_policy(counts)
        traj.append((s, pi, board.turn == chess.WHITE))

        legal_n = len(counts)
        stats["legal_sum"] += legal_n
        if legal_n > 0:
            s_counts = sum(counts.values())
            inv_s = 1.0 / (s_counts + 1e-9)
            ent = 0.0
            top1 = 0.0
            for c in counts.values():
                p = c * inv_s
                if p > top1:
                    top1 = p
                ent -= p * math.log(p + 1e-12)
            stats["entropy_sum"] += ent
            stats["top1_sum"] += top1
        stats["root_q_sum"] += root.Q()
        stats["plies"] += 1

        board.push(mv)
        ply += 1

    z_white = game_result_white(board)
    if resigned_result is not None:
        z_white = resigned_result
    stats["result"] = z_white
    stats["truncated"] = 0.0 if (board.is_game_over() or resigned_result is not None) else 1.0

    if resigned_result is None:
        z_white = shaped_game_value_white(board, truncated=not board.is_game_over())

    samples: List[Tuple[torch.Tensor, torch.Tensor, float]] = []
    for s, pi, to_play_white in traj:
        z = z_white if to_play_white else -z_white
        # augmentation (identity + mirror)
        for ss, pp in augment_sample(s, pi):
            samples.append((ss, pp, float(z)))
            stats["samples_n"] += 1
            stats["z_abs_sum"] += abs(z)
    return samples, stats


# =========================
# Training
# =========================
def train_step(
    net: PolicyValueNet, opt: torch.optim.Optimizer, batch: List[Tuple[torch.Tensor, torch.Tensor, float]]
) -> Dict[str, float]:
    net.train()
    x = torch.stack([b[0] for b in batch]).to(CFG.device)
    pi_t = torch.stack([b[1] for b in batch]).to(CFG.device)
    z_t = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=CFG.device)

    logits, v = net(x)

    logp = F.log_softmax(logits, dim=1)
    loss_p = -(pi_t * logp).sum(dim=1).mean()
    loss_v = F.mse_loss(v, z_t)
    loss = CFG.policy_loss_weight * loss_p + CFG.value_loss_weight * loss_v

    opt.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = float(nn.utils.clip_grad_norm_(net.parameters(), CFG.grad_clip))
    opt.step()

    with torch.no_grad():
        p_self_entropy = -(logp.exp() * logp).sum(dim=1).mean()
        p_self_top1 = logp.max(dim=1).values.exp().mean()
        pi_clamped = pi_t.clamp_min(1e-12)
        pi_entropy = -(pi_clamped * pi_clamped.log()).sum(dim=1).mean()
        pi_top1 = pi_t.max(dim=1).values.mean()
        v_mean = v.mean()
        v_std = v.std(unbiased=False)
        z_mean = z_t.mean()
        z_std = z_t.std(unbiased=False)

    return {
        "loss": float(loss.item()),
        "loss_p": float(loss_p.item()),
        "loss_v": float(loss_v.item()),
        "grad_norm": grad_norm,
        "p_ent": float(p_self_entropy.item()),
        "p_top1": float(p_self_top1.item()),
        "pi_ent": float(pi_entropy.item()),
        "pi_top1": float(pi_top1.item()),
        "v_mean": float(v_mean.item()),
        "v_std": float(v_std.item()),
        "z_mean": float(z_mean.item()),
        "z_std": float(z_std.item()),
    }


# =========================
# Eval vs baseline (GreedyBot)
# =========================
@torch.no_grad()
def net_pick_move_quick(net: PolicyValueNet, board: chess.Board) -> chess.Move:
    """
    Быстрый ход сети без MCTS (для оценки играем сеткой+MCTS или сеткой-быстро?).
    Для честности и силы лучше MCTS, но это долго.
    Компромисс: небольшой MCTS (меньше sims) на eval.
    """
    return chess.Move.null()


@torch.no_grad()
def play_net_vs_greedy(net: PolicyValueNet, net_is_white: bool, eval_sims: int = 32) -> float:
    """
    Возвращает результат партии в координатах 'net':
      +1 net победил, 0 ничья, -1 net проиграл
    """
    board = chess.Board()
    # временно уменьшим sims для eval (чтобы не убивать скорость)
    saved = CFG.mcts_sims
    CFG.mcts_sims = eval_sims
    try:
        ply = 0
        while not board.is_game_over() and ply < CFG.max_game_plies:
            net_turn = (board.turn == chess.WHITE) == net_is_white
            if net_turn:
                _, counts = run_mcts_batched(net, board)
                action = sample_from_counts(counts, temperature=1e-6)
                mv = action_to_move(action)
                if mv not in board.legal_moves:
                    mv = random.choice(list(board.legal_moves))
            else:
                mv = greedy_bot_move(board)

            board.push(mv)
            ply += 1

        # результат в координатах белых
        z_white = game_result_white(board)
        # перевод в координаты net
        if net_is_white:
            return z_white
        return -z_white
    finally:
        CFG.mcts_sims = saved


def elo_from_winrate(p: float) -> float:
    # относительный Elo (baseline Elo = 0) по формуле логита
    # clamp чтобы не улетать в бесконечность
    p = max(1e-6, min(1 - 1e-6, p))
    return 400.0 * math.log10(p / (1.0 - p))


@torch.no_grad()
def eval_vs_baseline(net: PolicyValueNet) -> Dict[str, float]:
    """
    Играем CFG.eval_games партий против GreedyBot, поровну цветами.
    Возвращаем статистику: win/draw/loss, score, elo_est
    """
    net.eval()
    wins = draws = losses = 0
    n = CFG.eval_games

    for i in range(n):
        net_is_white = (i % 2 == 0)
        r = play_net_vs_greedy(net, net_is_white=net_is_white, eval_sims=32)
        if r > 0.5:
            wins += 1
        elif r < -0.5:
            losses += 1
        else:
            draws += 1

    # score: win=1, draw=0.5
    score = wins + 0.5 * draws
    p = score / n  # expected score vs baseline
    elo = elo_from_winrate(p)

    return {
        "wins": float(wins),
        "draws": float(draws),
        "losses": float(losses),
        "score": float(score),
        "p": float(p),
        "elo": float(elo),
    }


# =========================
# Main loop
# =========================
def ensure_outdir():
    os.makedirs(CFG.out_dir, exist_ok=True)


def save_checkpoint(
    net: PolicyValueNet,
    opt: torch.optim.Optimizer,
    replay: ReplayBuffer,
    it: int,
    extra: Dict[str, float],
):
    path = os.path.join(CFG.out_dir, f"ckpt_iter_{it}.pt")
    ckpt = {
        "iter": it,
        "model_state": net.state_dict(),
        "opt_state": opt.state_dict(),
        "replay": {"max": replay.max, "pos": replay.pos, "buf": replay.buf},
        "rng": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
        },
        "config": CFG.__dict__,
        "extra": extra,
    }
    torch.save(ckpt, path)
    # also latest
    latest = os.path.join(CFG.out_dir, "latest.pt")
    torch.save(ckpt, latest)


def load_latest_checkpoint(
    net: PolicyValueNet,
    opt: torch.optim.Optimizer,
    replay: ReplayBuffer,
) -> int:
    latest = os.path.join(CFG.out_dir, "latest.pt")
    if not os.path.isfile(latest):
        return 0
    ckpt = torch.load(latest, map_location="cpu")
    net.load_state_dict(ckpt["model_state"])
    opt_state = ckpt.get("opt_state")
    if opt_state is not None:
        opt.load_state_dict(opt_state)
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(CFG.device)
    r = ckpt.get("replay")
    if r:
        replay.max = r.get("max", replay.max)
        replay.pos = r.get("pos", 0)
        replay.buf = r.get("buf", [])
    rng = ckpt.get("rng", {})
    if "python" in rng:
        random.setstate(rng["python"])
    if "torch" in rng:
        rng_t = rng["torch"]
        if torch.is_tensor(rng_t):
            if rng_t.device.type != "cpu":
                rng_t = rng_t.cpu()
            if rng_t.dtype != torch.uint8:
                rng_t = rng_t.byte()
            torch.set_rng_state(rng_t)
    return int(ckpt.get("iter", 0))


def main():
    print(f"Device: {CFG.device}")
    ensure_outdir()

    net = PolicyValueNet(channels=CFG.channels, blocks=CFG.res_blocks, action_size=CFG.action_size).to(CFG.device)
    opt = torch.optim.AdamW(net.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    replay = ReplayBuffer(CFG.replay_max)

    it = load_latest_checkpoint(net, opt, replay)
    while True:
        it += 1
        t0 = time.time()
        net.eval()

        # -------- self-play --------
        t_self0 = time.time()
        new_samples = 0
        sp_stats: List[Dict[str, float]] = []
        for g in range(CFG.games_per_iter):
            samples, st = self_play_game(net)
            for s, pi, z in samples:
                replay.add(s, pi, z)
            new_samples += len(samples)
            sp_stats.append(st)
            print(f"[iter {it}] selfplay {g+1}/{CFG.games_per_iter}: samples={len(samples)}")

        print(f"[iter {it}] replay size={len(replay)} (+{new_samples})")
        if sp_stats:
            games = len(sp_stats)
            plies = sum(s["plies"] for s in sp_stats)
            legal_sum = sum(s["legal_sum"] for s in sp_stats)
            ent_sum = sum(s["entropy_sum"] for s in sp_stats)
            top1_sum = sum(s["top1_sum"] for s in sp_stats)
            root_q_sum = sum(s["root_q_sum"] for s in sp_stats)
            samples_n = sum(s["samples_n"] for s in sp_stats)
            z_abs_sum = sum(s["z_abs_sum"] for s in sp_stats)
            trunc = sum(s["truncated"] for s in sp_stats)
            resigned = sum(s["resigned"] for s in sp_stats)
            wins = sum(1 for s in sp_stats if s["result"] > 0)
            draws = sum(1 for s in sp_stats if s["result"] == 0)
            losses = sum(1 for s in sp_stats if s["result"] < 0)

            avg_plies = plies / max(1, games)
            avg_legal = legal_sum / max(1, plies)
            avg_ent = ent_sum / max(1, plies)
            avg_top1 = top1_sum / max(1, plies)
            avg_root_q = root_q_sum / max(1, plies)
            sample_abs = z_abs_sum / max(1, samples_n)
            trunc_rate = trunc / max(1, games)
            resign_rate = resigned / max(1, games)

            print(
                f"[iter {it}] selfplay stats: W/D/L={wins}/{draws}/{losses} "
                f"trunc={trunc_rate:.2f} resign={resign_rate:.2f} avg_plies={avg_plies:.1f} "
                f"legal={avg_legal:.1f} ent={avg_ent:.3f} top1={avg_top1:.3f} "
                f"root_q={avg_root_q:.3f} z_abs={sample_abs:.3f}"
            )
        t_self = time.time() - t_self0

        # -------- training --------
        t_train0 = time.time()
        if len(replay) >= max(1000, CFG.batch_size):
            stats_hist: List[Dict[str, float]] = []
            for step in range(CFG.train_steps_per_iter):
                batch = replay.sample(CFG.batch_size)
                st = train_step(net, opt, batch)
                stats_hist.append(st)
                if (step + 1) % CFG.log_train_every == 0:
                    window = stats_hist[-CFG.log_train_every :]
                    avg = {k: sum(s[k] for s in window) / len(window) for k in window[0].keys()}
                    clip_frac = sum(1 for s in window if s["grad_norm"] > CFG.grad_clip) / len(window)
                    lr = opt.param_groups[0]["lr"]
                    print(
                        f"[iter {it}] train {step+1}/{CFG.train_steps_per_iter}: "
                        f"loss={avg['loss']:.4f} p={avg['loss_p']:.4f} v={avg['loss_v']:.4f} "
                        f"gn={avg['grad_norm']:.3f} clip={clip_frac:.2f} lr={lr:.2e} "
                        f"vμ={avg['v_mean']:.3f} vσ={avg['v_std']:.3f} "
                        f"zμ={avg['z_mean']:.3f} zσ={avg['z_std']:.3f} "
                        f"pi_ent={avg['pi_ent']:.3f} pi_top1={avg['pi_top1']:.3f} "
                        f"p_ent={avg['p_ent']:.3f} p_top1={avg['p_top1']:.3f}"
                    )
            if stats_hist:
                avg = {k: sum(s[k] for s in stats_hist) / len(stats_hist) for k in stats_hist[0].keys()}
                clip_frac = sum(1 for s in stats_hist if s["grad_norm"] > CFG.grad_clip) / len(stats_hist)
                lr = opt.param_groups[0]["lr"]
                print(
                    f"[iter {it}] train avg: loss={avg['loss']:.4f} p={avg['loss_p']:.4f} v={avg['loss_v']:.4f} "
                    f"gn={avg['grad_norm']:.3f} clip={clip_frac:.2f} lr={lr:.2e} "
                    f"vμ={avg['v_mean']:.3f} vσ={avg['v_std']:.3f} "
                    f"zμ={avg['z_mean']:.3f} zσ={avg['z_std']:.3f} "
                    f"pi_ent={avg['pi_ent']:.3f} pi_top1={avg['pi_top1']:.3f} "
                    f"p_ent={avg['p_ent']:.3f} p_top1={avg['p_top1']:.3f}"
                )
        else:
            print(f"[iter {it}] replay too small for training yet")
        t_train = time.time() - t_train0

        # -------- eval --------
        t_eval0 = time.time()
        extra = {}
        if it % CFG.eval_every_iters == 0 and len(replay) >= max(1000, CFG.batch_size):
            ev = eval_vs_baseline(net)
            extra.update(ev)
            print(
                f"[iter {it}] eval vs {CFG.baseline_name}: "
                f"W/D/L={int(ev['wins'])}/{int(ev['draws'])}/{int(ev['losses'])} "
                f"score={ev['score']:.1f}/{CFG.eval_games} p={ev['p']:.3f} elo~{ev['elo']:.1f}"
            )
        t_eval = time.time() - t_eval0

        # -------- save --------
        t_save0 = time.time()
        if it % CFG.save_every_iters == 0:
            save_checkpoint(net, opt, replay, it, extra)
        t_save = time.time() - t_save0

        dt = time.time() - t0
        print(
            f"[iter {it}] time: self={t_self:.1f}s train={t_train:.1f}s "
            f"eval={t_eval:.1f}s save={t_save:.1f}s total={dt:.1f}s\n"
        )


if __name__ == "__main__":
    main()
