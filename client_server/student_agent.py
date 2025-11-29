"""
Student Agent Implementation for River and Stones Game

This file contains the essential utilities and template for implementing your AI agent.
Your task is to complete the StudentAgent class with intelligent move selection.

Game Rules:
- Goal: Get 4 of your stones into the opponent's scoring area
- Pieces can be stones or rivers (horizontal/vertical orientation)  
- Actions: move, push, flip (stone↔river), rotate (river orientation)
- Rivers enable flow-based movement across the board

Your Task:
Implement the choose() method in the StudentAgent class to select optimal moves.
You may add any helper methods and modify the evaluation function as needed.
"""

import random
import copy
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import time


# ==================== GAME UTILITIES ====================
# Essential utility functions for game state analysis

class Weights:
    """
    This class is defined to hold the weights for evaluation function
    W1 : weight for feature -> number of pieces in scoring area
    W2 : weight for feature -> number of pieces attacking scoring area
    W3 : weight for feature -> number of pieces one move away from scoring area
    W4 : weight for feature -> manhattan distance of pieces
    W5 : weight for feature -> number of pieces defending scoring area
    """
    def __init__(self):
        self.W1 = 8
        self.W2 = 2
        self.W3 = 3
        self.W4 = 2
        self.W5 = 1

        self.penalty = 1
        self.last_distance = None

transposition_table = set()
def in_bounds(x: int, y: int, rows: int, cols: int) -> bool:
    """Check if coordinates are within board boundaries."""
    return 0 <= x < cols and 0 <= y < rows

def score_cols_for(cols: int) -> List[int]:
    """Get the column indices for scoring areas."""
    w = 4
    start = max(0, (cols - w) // 2)
    return list(range(start, start + w))

def top_score_row() -> int:
    """Get the row index for Circle's scoring area."""
    return 2

def bottom_score_row(rows: int) -> int:
    """Get the row index for Square's scoring area."""
    return rows - 3

def is_opponent_score_cell(x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> bool:
    """Check if a cell is in the opponent's scoring area."""
    if player == "circle":
        return (y == bottom_score_row(rows)) and (x in score_cols)
    else:
        return (y == top_score_row()) and (x in score_cols)

def is_own_score_cell(x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> bool:
    """Check if a cell is in the player's own scoring area."""
    if player == "circle":
        return (y == top_score_row()) and (x in score_cols)
    else:
        return (y == bottom_score_row(rows)) and (x in score_cols)

def get_opponent(player: str) -> str:
    """Get the opponent player identifier."""
    return "square" if player == "circle" else "circle"

# ==================== MOVE GENERATION HELPERS ====================

def generate_all_moves(board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
    """
    :param board:
    :param player:
    :param rows:
    :param cols:
    :param score_cols:
    :return: list of moves

    possible moves :
    1) move
    2) push
        a) stone to stone push
        b) river to stone push
    3) flip
    4) rotate
    """

    all_moves = []
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if not piece or piece.owner != player:
                continue

            if piece.side == "stone":
                # Move Stone
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if not in_bounds(nx, ny, rows, cols):
                        continue

                    # Check if player is moving into opponents scoring area
                    if is_opponent_score_cell(nx, ny, player, rows, cols, score_cols):
                        continue

                    target = board[ny][nx]

                    # Cell player is moving to is empty
                    if target is None:
                        all_moves.append({
                            "action": "move",
                            "from": [x, y],
                            "to": [nx, ny]
                        })

                    # Move through the river
                    elif target.side == "river":
                        flow_destinations = get_river_flow_destinations(
                            board, nx, ny, x, y, player, rows, cols, score_cols
                        )
                        for dest_x, dest_y in flow_destinations:
                            all_moves.append({
                                "action": "move",
                                "from": [x, y],
                                "to": [dest_x, dest_y]
                            })

                    else:
                        # Target is a stone - check if we can push it
                        px, py = nx + dx, ny + dy
                        if (in_bounds(px, py, rows, cols) and
                                board[py][px] is None and
                                not is_opponent_score_cell(px, py, target.owner, rows, cols, score_cols)):
                            all_moves.append({
                                "action": "push",
                                "from": [x, y],
                                "to": [nx, ny],
                                "pushed_to": [px, py]
                            })

                for orientation in ["horizontal", "vertical"]:
                    all_moves.append({
                        "action": "flip",
                        "from": [x, y],
                        "orientation": orientation
                    })

            else:  # piece.side == "river"
                # Player moving a river
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if not in_bounds(nx, ny, rows, cols):
                        continue

                    # Cannot move into opponent's score area
                    if is_opponent_score_cell(nx, ny, player, rows, cols, score_cols):
                        continue

                    target = board[ny][nx]

                    if target is None:
                        # Simple move to empty cell
                        all_moves.append({
                            "action": "move",
                            "from": [x, y],
                            "to": [nx, ny]
                        })

                    elif target.side == "river":
                        # Move through river flow
                        flow_destinations = get_river_flow_destinations(
                            board, nx, ny, x, y, player, rows, cols, score_cols
                        )
                        for dest_x, dest_y in flow_destinations:
                            all_moves.append({
                                "action": "move",
                                "from": [x, y],
                                "to": [dest_x, dest_y]
                            })

                    else:
                        # Target is a stone - river can push with flow
                        flow_destinations = get_river_flow_destinations(
                            board, nx, ny, x, y, target.owner, rows, cols, score_cols, river_push=True
                        )
                        for dest_x, dest_y in flow_destinations:
                            if not is_opponent_score_cell(dest_x, dest_y, target.owner, rows, cols, score_cols):
                                all_moves.append({
                                    "action": "push",
                                    "from": [x, y],
                                    "to": [nx, ny],
                                    "pushed_to": [dest_x, dest_y]
                                })

                # Flipping a river
                all_moves.append({
                    "action": "flip",
                    "from": [x, y]
                })

                # rotating a river
                all_moves.append({
                    "action": "rotate",
                    "from": [x, y]
                })

    return all_moves


def get_river_flow_destinations(board: List[List[Any]], rx: int, ry: int, sx: int, sy: int,
                                player: str, rows: int, cols: int, score_cols: List[int],
                                river_push: bool = False) -> List[Tuple[int, int]]:
    """
    :param board:
    :param rx:
    :param ry:
    :param sx:
    :param sy:
    :param player:
    :param rows:
    :param cols:
    :param score_cols:
    :param river_push:
    :return: list of tuples with river orientation
    """
    destinations = []
    visited = set()
    queue = [(rx, ry)]

    while queue:
        x, y = queue.pop(0)
        if (x, y) in visited or not in_bounds(x, y, rows, cols):
            continue
        visited.add((x, y))

        cell = board[y][x]
        if river_push and x == rx and y == ry:
            cell = board[sy][sx]

        if cell is None:
            if is_opponent_score_cell(x, y, player, rows, cols, score_cols):
                # Block entering opponent score
                pass
            else:
                destinations.append((x, y))
            continue

        if cell.side != "river":
            continue

        # River flow directions
        dirs = [(1, 0), (-1, 0)] if cell.orientation == "horizontal" else [(0, 1), (0, -1)]

        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            while in_bounds(nx, ny, rows, cols):
                if is_opponent_score_cell(nx, ny, player, rows, cols, score_cols):
                    break

                next_cell = board[ny][nx]
                if next_cell is None:
                    destinations.append((nx, ny))
                    nx += dx
                    ny += dy
                    continue

                if nx == sx and ny == sy:
                    nx += dx
                    ny += dy
                    continue

                if next_cell.side == "river":
                    queue.append((nx, ny))
                    break
                break

    # Remove duplicates
    unique_destinations = []
    seen = set()
    for d in destinations:
        if d not in seen:
            seen.add(d)
            unique_destinations.append(d)

    return unique_destinations

def count_stones_near_scoring_area(board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int], defend : bool) -> int:
    """
    :param board:
    :param player:
    :param rows:
    :param cols:
    :param score_cols:
    :param defend : if True then we will calculate the pieces which are defending opponent's scoring area
    :return: if rev returns the number of player's pieces near its scoring area (does include the ones in scoring area)
             else returns the number of player's pieces near opponents scoring area
    """
    count = 0

    if player == "circle":
        if not defend:
            score_row = top_score_row()
        else:
            score_row = bottom_score_row(rows)
    else:
        if not defend:
            score_row = bottom_score_row(rows)
        else:
            score_row = top_score_row()

    radius = 1
    row_lower_bound = score_row - radius
    row_upper_bound = score_row + radius
    col_lower_bound = score_cols[0] - radius
    col_upper_bound = score_cols[-1] + radius

    for y in range(col_lower_bound, col_upper_bound + 1):
        for x in range(row_lower_bound, row_upper_bound + 1):
            piece = board[x][y]
            if piece and piece.owner == player and piece.side == "stone":
                count += 1
    return count / 12.0

def count_stones_in_scoring_area(board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> int:
    """
    :param board:
    :param player:
    :param rows:
    :param cols:
    :param score_cols:
    :return: number of pieces in scoring area
    """
    count = 0
    if player == "circle":
        score_row = top_score_row()
    else:
        score_row = bottom_score_row(rows)

    for x in score_cols:
        if in_bounds(x, score_row, rows, cols):
            piece = board[score_row][x]
            if piece and piece.owner == player and piece.side == "stone":
                count += 1

    return count / 12.0

def compute_valid_targets(board:List[List[Optional[Any]]],
                          sx:int, sy:int, player:str,
                          rows:int, cols:int, score_cols:List[int]) -> Dict[str,Any]:
    if not in_bounds(sx,sy,rows,cols):
        return {'moves': set(), 'pushes': []}
    p = board[sy][sx]
    if p is None or p.owner != player:
        return {'moves': set(), 'pushes': []}
    moves=set(); pushes=[]
    dirs=[(1,0),(-1,0),(0,1),(0,-1)]
    for dx,dy in dirs:
        tx,ty = sx+dx, sy+dy
        if not in_bounds(tx,ty,rows,cols): continue
        # block entering opponent score cell
        if is_opponent_score_cell(tx,ty,player,rows,cols,score_cols):
            continue
        target = board[ty][tx]
        if target is None:
            moves.add((tx,ty))
        elif target.side == "river":
            flow = get_river_flow_destinations(board, tx, ty, sx, sy, player, rows, cols, score_cols)
            for d in flow: moves.add(d)
        else:
            # stone occupied
            if p.side == "stone":
                px,py = tx+dx, ty+dy
                if in_bounds(px,py,rows,cols) and board[py][px] is None and not is_opponent_score_cell(px,py,p.owner,rows,cols,score_cols):
                    pushes.append(((tx,ty),(px,py)))
            else:
                pushed_player = target.owner
                flow = get_river_flow_destinations(board, tx, ty, sx, sy, pushed_player, rows, cols, score_cols, river_push=True)
                for d in flow:
                    if not is_opponent_score_cell(d[0],d[1],player,rows,cols,score_cols):
                        pushes.append(((tx,ty),(d[0],d[1])))
    return {'moves': moves, 'pushes': pushes}

def count_reachable_in_one(board:List[List[Optional[Any]]],
                           player:str, rows:int, cols:int, score_cols:List[int]) -> int:
    """
    m_self: number of player's pieces (stone side up) that can reach the player's scoring
    area in one legal move (including moves produced by river flow and pushes returned
    by compute_valid_targets).
    """
    m = 0
    for y,row in enumerate(board):
        for x,p in enumerate(row):
            if p and p.owner == player and p.side == "stone":
                if is_own_score_cell(x, y, player, rows, cols, score_cols):
                    continue
                info = compute_valid_targets(board, x, y, player, rows, cols, score_cols)
                # moves is a set of (tx,ty)
                for (tx,ty) in info.get('moves', set()):
                    if is_own_score_cell(tx, ty, player, rows, cols, score_cols):
                        m += 1
                        break
                else:
                    # check pushes: pushes is list of ((tx,ty),(ptx,pty))
                    for of,pushed in info.get('pushes', []):
                        ptx, pty = pushed
                        if is_own_score_cell(ptx, pty, player, rows, cols, score_cols):
                            m += 1
                            break
    return m / 12.0

def count_manhattan_distance(board:List[List[Optional[Any]]],
                           player:str, rows:int, cols:int, score_cols:List[int]) -> int:
    if player == "circle":
        score_row = top_score_row()
    else:
        score_row = bottom_score_row(rows)
    total_dist = 0
    number_of_pieces = 0
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if piece and piece.side == "stone":
                if piece.owner == player:
                    number_of_pieces += 1
                    vertical_distance = abs(score_row - x)
                    if y in score_cols:
                        total_dist += vertical_distance
                    else:
                        if y < min(score_cols):
                            total_dist += (vertical_distance + abs(min(score_cols) - y))
                        else:
                            total_dist += (vertical_distance + abs(max(score_cols) - y))
    avg_man_dist = total_dist / 12
    max_dist = rows + cols
    norm_man_dist = avg_man_dist / max_dist
    return norm_man_dist

def board_hash(board):
    return tuple(
        tuple((p.owner, p.side, getattr(p, "orientation", None)) if p else None for p in row)
        for row in board
    )

def basic_evaluate_board(board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int], weights : Weights) -> float:
    """
    :param board:
    :param player:
    :param rows:
    :param cols:
    :param score_cols:
    :param weights:
    :return: score of the board
    """
    score = 0.0
    opponent = get_opponent(player)
    
    # Pieces in Scoring Area
    players_pieces_in_scoring_area = count_stones_in_scoring_area(board, player, rows, cols, score_cols)
    opponents_pieces_in_scoring_area = count_stones_in_scoring_area(board, opponent, rows, cols, score_cols)
    #print(players_pieces_in_scoring_area, opponents_pieces_in_scoring_area)

    # Get the number of pieces attacking near scoring area for both players
    players_pieces_attacking_scoring_area = count_stones_near_scoring_area(board, player, rows, cols, score_cols,
                                                                           False) - players_pieces_in_scoring_area
    opponents_pieces_attacking_scoring_area = count_stones_near_scoring_area(board, opponent, rows, cols, score_cols,
                                                                             False) - opponents_pieces_in_scoring_area
    #print(players_pieces_attacking_scoring_area, opponents_pieces_attacking_scoring_area)
    # Stones that can be moved in one single move
    players_single_move_scorer = count_reachable_in_one(board, player, rows, cols, score_cols)
    opponents_single_move_scorer = count_reachable_in_one(board, opponent, rows, cols, score_cols)
    #print(players_single_move_scorer, opponents_single_move_scorer)

    # Calculate manhattan distance
    player_manhattan_distance = count_manhattan_distance(board, player, rows, cols, score_cols)
    opponent_manhattan_distance = count_manhattan_distance(board, opponent, rows, cols, score_cols)
    #print(player_manhattan_distance, opponent_manhattan_distance)

    # Get the number of pieces defending near scoring area for both players
    players_pieces_defending_scoring_area = count_stones_near_scoring_area(board, player, rows, cols, score_cols,
                                                                           True)
    opponents_pieces_defending_scoring_area = count_stones_near_scoring_area(board, opponent, rows, cols, score_cols,
                                                                             True)
    #print(players_pieces_defending_scoring_area, opponents_pieces_defending_scoring_area)

    F1 = players_pieces_in_scoring_area - opponents_pieces_in_scoring_area
    F2 = players_pieces_attacking_scoring_area - opponents_pieces_attacking_scoring_area
    F3 = players_single_move_scorer - opponents_single_move_scorer
    F4 = opponent_manhattan_distance - player_manhattan_distance
    F5 = players_pieces_defending_scoring_area - opponents_pieces_defending_scoring_area

    score = weights.W1*F1 + weights.W2*F2 + weights.W3*F3 + weights.W4*F4 + weights.W5*F5

    # Adding to hash set if not seen or giving a penalty
    state = board_hash(board)
    if state in transposition_table:
        score -= weights.penalty
    else:
        transposition_table.add(state)

    return score

def simulate_move(board: List[List[Any]], move: Dict[str, Any], player: str, rows: int, cols: int, score_cols: List[int]) -> Tuple[bool, Any, Any]:
    """
    :param board:
    :param move:
    :param player:
    :param rows:
    :param cols:
    :param score_cols:
    :return: success, new_board
    """
    try:
        from gameEngine import validate_and_apply_move
        my, mx = move['from']
        moving_piece = copy.deepcopy(board[mx][my])
        success, message = validate_and_apply_move(board, move, player, rows, cols, score_cols)
        return success, board, moving_piece if success else message
    except ImportError as e:
        print(e)
        # Fallback to basic simulation if game engine not available
        return True, copy.deepcopy(board), None

def revert_move(board: List[List[Any]], move: Dict[str, Any], player: str, piece: Any, rows: int, cols: int, score_cols: List[int]) -> Tuple[bool, Any]:
    """
    :param board:
    :param move:
    :param player:
    :param rows:
    :param cols:
    :param score_cols:
    :return: board with move undoed

    Types of moves to undo

    1) MOVE : (fy, fx) -> (ty, tx)
        * stone : (ty, tx) -> (fy, fx)
        * river : (ty, tx) -> (fy, fx)
    2) PUSH : (fy, fx) -> (ty, tx) and (ty, tx) -> (pty, ptx)
        a) stone pushing stone
            * pushing stone : (ty, tx) -> (fy, fx)
            * pushed stone : (pty,ptx) -> (ty, tx)
        b) river pushing river (will have to remember orientation of river)
            * river : (ty, tx) -> (fy, fx) and orientation = previous orientation
            * stone :  (pty,ptx) -> (ty, tx)
    3) FLIP :
        a) stone flipped : flip back and clear orientation
        b) river flipped : remember orientation, flip back and set that orientation
    4) ROTATE : orientation = "horizontal" if orientation == "vertical" else orientation == "vertical"
    """
    try:
        from gameEngine import validate_and_apply_move
        if move['action'] == 'move':
            new_move = {
                "action": "move",
                "from": move['to'],
                "to": move['from']
            }
            if piece.side == "stone" or piece.side == "river":
                success, message = validate_and_apply_move(board, new_move, player, rows, cols, score_cols)
                return (success, message)

        elif move['action'] == "push":
            fy, fx = move['from']
            ty, tx = move['to']
            pty, ptx = move['pushed_to']
            # print(f"moving pushing pieces from : {ty, tx} to {fy, fx}")
            # print(f"moving pushed pieces from : {pty, ptx} to {ty, tx}")
            pushed_piece = board[ptx][pty]
            board[fx][fy] = piece
            board[tx][ty] = pushed_piece
            board[ptx][pty] = None
            return (True, "Moved back all pieces")

        elif move['action'] == 'flip':
            fy, fx = move['from']
            # print(f"Flipped back piece at {fy, fx} orientation {piece.orientation}")
            board[fx][fy] = piece
            return (True, "Moved back all pieces")

        elif move['action'] == 'rotate':
            fy, fx = move['from']
            # print(f"Flipped back piece at {fy, fx} orientation {piece.orientation}")
            board[fx][fy] = piece
            return (True, "Moved back all pieces")

        else:
            return False, "No movement Possible"

    except ImportError:
        # Fallback to basic simulation if game engine not available
        return True, copy.deepcopy(board)

# ==================== BASE AGENT CLASS ====================
def minimax(board: List[List[Any]], depth: int, alpha: float, beta: float, maximizing: bool, player: str, rows: int, cols: int, score_cols:List[int], weights: Weights):
    if depth == 0:
        return basic_evaluate_board(board, player, rows, cols, score_cols, weights), None

    current_player = player if maximizing else get_opponent(player)
    moves = generate_all_moves(board, current_player, rows, cols, score_cols)

    if not moves:
        return basic_evaluate_board(board, player, rows, cols, score_cols, weights), None

    best_move = None

    if maximizing:
        max_eval = float("-inf")
        for move in moves:
            success, board, piece = simulate_move(board, move, current_player, rows, cols, score_cols)
            if not success:
                continue
            eval_score, _ = minimax(board, depth-1, alpha, beta, False, player, rows, cols, score_cols, weights)
            if move['action'] == "flip":
                eval_score -= weights.penalty
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            revert_move(board, move, current_player, piece, rows, cols, score_cols)
            if beta <= alpha:
                break
        return max_eval, best_move

    else:
        min_eval = float("inf")
        for move in moves:
            success, board, piece = simulate_move(board, move, current_player, rows, cols, score_cols)
            if not success:
                continue
            eval_score, _ = minimax(board, depth-1, alpha, beta, True, player, rows, cols, score_cols, weights)
            if move['action'] == "flip":
                eval_score -= weights.penalty
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            revert_move(board, move, current_player, piece, rows, cols, score_cols)
            if beta <= alpha:
                break
        return min_eval, best_move


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    """
    
    def __init__(self, player: str):
        """Initialize agent with player identifier."""
        self.player = player
        self.opponent = get_opponent(player)
    
    @abstractmethod
    def choose(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int], current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        """
        Choose the best move for the current board state.
        
        Args:
            board: 2D list representing the game board
            rows, cols: Board dimensions
            score_cols: List of column indices for scoring areas
        
        Returns:
            Dictionary representing the chosen move, or None if no moves available
        """
        pass

# ==================== STUDENT AGENT IMPLEMENTATION ====================

class StudentAgent(BaseAgent):
    """
    Student Agent Implementation
    
    TODO: Implement your AI agent for the River and Stones game.
    The goal is to get 4 of your stones into the opponent's scoring area.
    
    You have access to these utility functions:
    - generate_all_moves(): Get all legal moves for current player
    - basic_evaluate_board(): Basic position evaluation 
    - simulate_move(): Test moves on board copy
    - count_stones_in_scoring_area(): Count stones in scoring positions
    """
    
    def __init__(self, player: str):
        super().__init__(player)
        if player == "circle":
            print("CIRCLE")
            top_left_y, top_left_x = [3, 8]
            downMove = 1
            top_corner_x = 0

        else:
            print("SQUARE")
            top_left_y, top_left_x = [3, 4]
            downMove = -1
            top_corner_x = 12

        self.not_normal = False
        self.return_to_normal_in = 5
        self.previous_pos = None
        self.looping_moves = 0
        self.weights = Weights()
        self.move_no = 0
        self.start_move_set = {
            1: {"action": "push", 'from': [top_left_y, top_left_x], 'to': [top_left_y, top_left_x + downMove], 'pushed_to': [top_left_y, top_left_x + 2 * downMove]},
            2: {"action": "push", 'from': [top_left_y + 5, top_left_x], 'to': [top_left_y + 5, top_left_x + downMove], 'pushed_to': [top_left_y + 5, top_left_x + 2 * downMove]},
            3: {"action": "flip", 'from': [top_left_y + 1, top_left_x], 'orientation': 'horizontal'},
            4: {"action": "flip", 'from': [top_left_y + 2, top_left_x], 'orientation': 'horizontal'},
            5: {"action": "flip", 'from': [top_left_y + 3, top_left_x], 'orientation': 'vertical'},
            6: {"action": "flip", 'from': [top_left_y + 4, top_left_x], 'orientation': 'vertical'},
        }
        # self.aggressive_move_set = {
        #     7: {"action": "move", "from": [top_left_y + 3, top_left_x], "to": [0, top_left_x]},
        #     8: {"action": "move", "from": [top_left_y + 2, top_left_x], "to": [0, top_corner_x]},
        #     9: {"action": "move", "from": [top_left_y + 4, top_left_x], "to": [top_left_y + 3, top_left_x]},
        #     10: {"action": "move", "from": [top_left_y + 3, top_left_x], "to": [top_left_y + 2, top_left_x]},
        #     11: {"action": "move", "from": [top_left_y + 2, top_left_x], "to": [5, top_corner_x]},
        # }
        # TODO: Add any initialization you need
    
    def choose(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int], current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        """
        Choose the best move for the current board state.
        
        Args:
            board: 2D list representing the game board
            rows, cols: Board dimensions  
            score_cols: Column indices for scoring areas
            
        Returns:
            Dictionary representing your chosen move
        """
        move = None
        if self.move_no < 6:
            self.move_no += 1
            move = self.start_move_set[self.move_no]
        else:
            # print("SELECTING MOVES FROM HERE FOR ", self.player)
            depth = 2  # start small; increase if performance allows
            _, best_move = minimax(board, depth, float("-inf"), float("inf"), True, self.player, rows, cols, score_cols, self.weights)
            move = best_move

        if self.not_normal:
            self.return_to_normal_in -= 1
        if self.return_to_normal_in == 0:
            self.return_to_normal_in = 5
            self.not_normal = False
            self.weights.W4 = 2

        if self.previous_pos is not None and move['action'] not in ['flip', 'rotate'] and self.previous_pos == move['to']:
            self.looping_moves += 1
        if self.looping_moves >= 5:
            self.not_normal = True
            self.weights.W4 = 5
            self.looping_moves = 0

        self.previous_pos = move['from']
        # print(best_move, max_score)
        return move
        # TODO: Replace random selection with your AI algorithm

# ==================== TESTING HELPERS ====================
def test_student_agent():
    """
    Basic test to verify the student agent can be created and make moves.
    """
    print("Testing StudentAgent...")
    
    try:
        from gameEngine import default_start_board, DEFAULT_ROWS, DEFAULT_COLS, board_to_ascii
        
        rows, cols = DEFAULT_ROWS, DEFAULT_COLS
        score_cols = score_cols_for(cols)
        board = default_start_board(rows, cols)
        board_copy = copy.deepcopy(board)
        agent = StudentAgent("circle")
        result = board_to_ascii(board, rows, cols, score_cols)
        print(result)
        moves = generate_all_moves(board, 'circle', rows, cols, score_cols)
        from gameEngine import  validate_and_apply_move
        move_no = 1

        for move in moves:
            success, board, piece = simulate_move(board, move, 'circle', rows, cols, score_cols)
            if success:
                print(f"For move {move_no}: ", move, " We get : ", success)
                move_no += 1
                result = board_to_ascii(board, rows, cols, score_cols)
                print(result)
                revert_move(board, move, 'circle', piece, rows, cols, score_cols)
                result = board_to_ascii(board, rows, cols, score_cols)
                print(result)
            else:
                print("THE MOVE WAS NOT VALID")
        if board_copy == board:
            print("NO CHANGES")
        else:
            print("OH OOH YOU FUCKED UP")
        print("LENGTH OF MOVES ", len(moves))
       #  move = agent.choose(board, rows, cols, score_cols,1.0,1.0)
        
        if move:
            print("✓ Agent successfully generated a move")
        else:
            print("✗ Agent returned no move")
    
    except ImportError:
        agent = StudentAgent("circle")
        print("✓ StudentAgent created successfully")

if __name__ == "__main__":
    # Run basic test when file is executed directly
    test_student_agent()
