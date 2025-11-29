#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <map>
#include <random>
#include <chrono>
#include <stack>
#include <set>


using namespace std;
#include <queue>
#include <set>
#include <algorithm>  // for find
#include <iostream>
namespace py = pybind11;
using std::vector;
using std::pair;
using std::set;
using std::queue;
using std::string;

/*
=========================================================
 STUDENT AGENT FOR STONES & RIVERS GAME
---------------------------------------------------------
 The Python game engine passes the BOARD state into C++.
 Each board cell is represented as a dictionary in Python:

    {
        "owner": "circle" | "square",          // which player owns this piece
        "side": "stone" | "river",             // piece type
        "orientation": "horizontal" | "vertical"  // only relevant if side == "river"
    }

 In C++ with pybind11, this becomes:

    Board

 Meaning:
   - board[y][x] gives the cell at (x, y).
   - board[y][x].empty() → true if the cell is empty (no piece).
   - board[y][x].at("owner") → "circle" or "square".
   - board[y][x].at("side") → "stone" or "river".
   - board[y][x].at("orientation") → "horizontal" or "vertical".

=========================================================
*/

// ---- Move struct ----
struct Move {
    string action;
    vector<int> from;
    vector<int> to;
    vector<int> pushed_to;
    string orientation;
};

// class to store tree node and childrens
struct Node {
    int id;
    string player;
    int level;
    float score;
    Move move;
    Node* parent;
    vector<Node*> children;
    Node(int id_ = -1,
         string player_ = "",
         int level_ = 0,
         float score_ = 0.0,
         Move move_ = Move(),
         Node* parent_ = nullptr)
        : id(id_), player(player_), level(level_), score(score_), move(move_), parent(parent_) {}
};

struct Board {
    vector<vector<map<string, string>>> cells;
    Board() = default;
    Board(const vector<vector<map<string, string>>>& b) : cells(b) {}
    vector<map<string, string>>& operator[](size_t i) {
        return cells[i];
    }

    const vector<map<string, string>>& operator[](size_t i) const {
        return cells[i];
    }
};


// function to make move, input board and move, output board with move
Board makemove(
    const Board& board,
    const Move& move,
    const string& player,
    int rows, int cols,
    const vector<int>& score_cols
);
// function to reverse move, input board and move, output board with move
Board reversemove(Board& board,Move move);

// function to return tree or first find tree then do minmax
Move IDS(Board& board, int row, int col, const vector<int>& score_cols, float current_player_time, float opponent_time, string side, int depth );
// ---- Declarations -----
vector<Move> generate_all_moves(const Board& board, int row, int col, const vector<int>& score_cols, float current_player_time, float opponent_time, const string& player);


double randfgen(){
    random_device rd;   // Non-deterministic random seed
    mt19937 gen(rd());  // Mersenne Twister generator
    uniform_real_distribution<double> dist(0.0, 1.0); // [0,1)

    return dist(gen);
}

void print_board(const Board& board) {
    int rows = board.cells.size();
    int cols = board.cells[0].size();

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            const auto &cell = board[y][x];
            if (cell.empty()) {
                cout << ". "; // empty cell
            } else {
                string type = cell.at("side"); // assuming "side" is stone or river
                if (type == "stone") {
                    cout << "s ";
                } else if (type == "river") {
                    string orient = cell.at("orientation");
                    if (orient == "horizontal") {
                        cout << "h ";
                    } else if (orient == "vertical") {
                        cout << "v ";
                    } else {
                        cout << "? "; // unknown orientation
                    }
                } else {
                    cout << "? "; // unknown type
                }
            }
        }
        cout << endl;
    }
}
Move defence(Board& board, int rows, int cols, const vector<int>& score_cols, string& player){
    // defence move logic: don't move first row of 6 stones, make end 2 rivers to avoid push
    
    // check if defence already done
    int top_score_row = 2;
    int bottom_score_row = rows-3;
    // cout << "defence of player " << player << endl;
    int mydefenserow = player == "circle" ? bottom_score_row - 1 : top_score_row + 1;
    // for (int x = 0; x < int(cols/2); x++) {
    //     const auto& cell = board[mydefenserow][x+3];
    //     if (!cell.empty()){
    //         if (cell.at("side") == "stone" && cell.at("owner") == player) {
    //             Move move = Move({"flip", {x+3, mydefenserow}, {}, {}, "horizontal"});
    //             return move;
    //         }
    //     }
    // }
    const auto& cell = board[mydefenserow][0+3];
    if (!cell.empty()){
        if (cell.at("side") == "stone" && cell.at("owner") == player) {
            Move move = Move({"flip", {0+3, mydefenserow}, {}, {}, "horizontal"});
            return move;
        }
    }
    int x = int(cols/2)+2;
    const auto& cell2 = board[mydefenserow][x];
    if (!cell2.empty()){
        if (cell2.at("side") == "stone" && cell2.at("owner") == player) {
            Move move = Move({"flip", {int(cols/2)+2, mydefenserow}, {}, {}, "horizontal"});
            return move;
        }
    }

    return Move({"invalid", {}, {}, {}, ""});
}

// ---- Student Agent ----
class StudentAgent {
public:
    explicit StudentAgent(string side) : side(move(side)), gen(rd()) {}

    Move choose(const  vector<vector<map<string, string>>>& oldboard, int row, int col, const vector<int>& score_cols, float current_player_time, float opponent_time, bool flag) {
        Board board = Board(oldboard);
        int depth =3;
        // cout << "CPP Agent choosing move..." << endl;
        if (current_player_time <= 3 or flag==false) {
            vector<Move> moves = generate_all_moves(board, row, col, score_cols, current_player_time, opponent_time, side);
            if (moves.empty()) {
                return {"move", {0,0}, {0,0}, {}, ""}; // fallback
            }
            uniform_int_distribution<> dist(0, moves.size()-1);
            return moves[dist(gen)];
        }   
        if (current_player_time < 22){
            depth =2;
            Move move = IDS(board,row,col,score_cols,current_player_time,opponent_time,side,depth);
            return move;
        }
        
        // calling IDS
        auto start = chrono::high_resolution_clock::now();
        // cout << "calling ids... " << endl;
        // cout << "time left (current player): " << current_player_time << " s, (opponent player): " << opponent_time << endl;
        // defence setup first 
        Move move = defence(board,row,col,score_cols,side);
        if (move.action != "invalid"){
            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
            // cout << "Defence move " << move.action  << " at " << move.from[0] << ", " << move.from[1] << " chosen in " << duration << " ms" << endl;
            return move;
        }

        // this_thread::sleep_for(chrono::seconds(3));
        double rf = randfgen();
        
        if (rf<0.3){
            depth =1;
        }
        else if (rf<0.8){
            depth =2;
        }
        // cout << "IDS depth chosen: " << depth << endl;
        // print_board(board);
        move = IDS(board,row,col,score_cols,current_player_time,opponent_time,side,depth);
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        // cout << "IDS took " << duration << " ms for depth: " << depth << endl;
        if (move.action != ""){
            return move;
        }
        
        vector<Move> moves = generate_all_moves(board, row, col, score_cols, current_player_time, opponent_time, side);

        uniform_int_distribution<> dist(0, moves.size()-1);
        return moves[dist(gen)];
    
    }   


private:
    string side;
    random_device rd;
    mt19937 gen;
};


// ---- Functions manual implementation

// Helper functions
bool in_bounds(int x, int y, int rows, int cols) {
    return x >= 0 && x < cols && y >= 0 && y < rows;
}

bool is_opponent_score_cell(int x, int y, const string& player,
                            int rows, int cols, const vector<int>& score_cols) {
    if (player == "circle") {
        // bool res = (y == rows - 3) && (find(score_cols.begin(), score_cols.end(), x) != score_cols.end());
        // cout << "inside hceck score fn, circle" << res << " " << x << "," << y << endl;
        return (y == rows - 3) && (find(score_cols.begin(), score_cols.end(), x) != score_cols.end());
    } else {
        // bool res = (y == 2) && (find(score_cols.begin(), score_cols.end(), x) != score_cols.end());
        // cout << "inside hceck score fn butsquare," << res  << " " << x << "," << y << endl;
        return (y == 2) && (find(score_cols.begin(), score_cols.end(), x) != score_cols.end());
    }
}

bool is_my_score_cell(int x, int y, const string& player,
                            int rows, int cols, const vector<int>& score_cols) {
    if (player != "circle") {
        // bool res = (y == rows - 3) && (find(score_cols.begin(), score_cols.end(), x) != score_cols.end());
        // cout << "inside hceck score fn, circle" << res << " " << x << "," << y << endl;
        return (y == rows - 3) && (find(score_cols.begin(), score_cols.end(), x) != score_cols.end());
    } else {
        // bool res = (y == 2) && (find(score_cols.begin(), score_cols.end(), x) != score_cols.end());
        // cout << "inside hceck score fn butsquare," << res  << " " << x << "," << y << endl;
        return (y == 2) && (find(score_cols.begin(), score_cols.end(), x) != score_cols.end());
    }
}

int get_score(const Board& board, const string& player,
              int rows, int cols, const vector<int>& score_cols) {
    int score = 0;
    int score_row = (player == "circle") ? (rows - 3) : 2;

    for (int x : score_cols) {
        auto cell = board[score_row][x];
        if (!cell.empty() && cell.at("owner") == player) {
            score++;
        }
    }
    return score;
}

vector<pair<int,int>> get_river_flow_destinations(
    const Board& board,
    int rx, int ry, int sx, int sy, const string& player,
    int rows, int cols, const vector<int>& score_cols,
    bool river_push = false
) {
    vector<pair<int,int>> destinations;
    set<pair<int,int>> visited;
    queue<pair<int,int>> q;
    q.push({rx, ry});

    while (!q.empty()) {
        auto cur = q.front(); q.pop();
        int x = cur.first, y = cur.second;
        if (visited.count(cur) || !in_bounds(x, y, rows, cols)) continue;
        visited.insert(cur);

        // emulate Python's "if river_push and x==rx and y==ry: cell = board[sy][sx]"
        auto cell = board[y][x];
        bool using_replaced_cell = (river_push && x == rx && y == ry);
        if (using_replaced_cell) {
            // treat as if cell == board[sy][sx]
            // we'll refer to board[sy][sx] via next_cell checks below where needed
        }

        // empty cell case
        if (!using_replaced_cell && cell.empty()) {
            if (!is_opponent_score_cell(x, y, player, rows, cols, score_cols)) {
                destinations.push_back({x, y});
            }
            continue;
        }

        // if we replaced the cell with pushed player's cell:
        if (using_replaced_cell) {
            // fall through to river-check using board[sy][sx]
            // make cell refer to that one
            // (we already have board in scope)
            // we don't mutate `cell` variable because it's const-like; we'll access board[sy][sx]
        }

        // ensure we have a river here
        const auto& real_cell = using_replaced_cell ? board[sy][sx] : cell;
        if (real_cell.empty()) {
            // already handled above; keep parity with python
            if (!is_opponent_score_cell(x, y, player, rows, cols, score_cols)) {
                destinations.push_back({x, y});
            }
            continue;
        }
        if (real_cell.at("side") != "river") continue;

        // orientation-driven directions
        vector<pair<int,int>> dirs;
        if (real_cell.at("orientation") == "horizontal") {
            dirs = {{1,0},{-1,0}};
        } else {
            dirs = {{0,1},{0,-1}};
        }

        for (auto d : dirs) {
            int dx = d.first, dy = d.second;
            int nx = x + dx, ny = y + dy;
            while (in_bounds(nx, ny, rows, cols)) {
                if (is_opponent_score_cell(nx, ny, player, rows, cols, score_cols)) break;

                // next_cell analogous to Python's next_cell = board[ny][nx]
                const auto& next_cell = board[ny][nx];

                // IMPORTANT: keep same ordering as Python:
                // 1) if next_cell empty -> add and continue
                if (next_cell.empty()) {
                    destinations.push_back({nx, ny});
                    nx += dx; ny += dy;
                    continue;
                }

                // 2) if it's exactly the source square (sx,sy) skip over it and continue scanning
                if (nx == sx && ny == sy) {
                    nx += dx; ny += dy;
                    continue;
                }

                // 3) if that next_cell is river, enqueue and stop scanning this direction
                if (next_cell.at("side") == "river") {
                    q.push({nx, ny});
                    break;
                }

                // otherwise blocked by a non-river occupied cell
                break;
            }
        }
    }

    // dedupe while preserving order (like Python)
    set<pair<int,int>> seen;
    vector<pair<int,int>> out;
    out.reserve(destinations.size());
    for (auto &d : destinations) {
        if (!seen.count(d)) { seen.insert(d); out.push_back(d); }
    }
    return out;
}


vector<Move> generate_all_moves(
    const Board& board,
    int rows, int cols,
    const vector<int>& score_cols,
    float /*current_player_time*/,
    float /*opponent_time*/,
    const string& player
) {
    vector<Move> moves;
    vector<pair<int,int>> dirs = {{1,0},{-1,0},{0,1},{0,-1}};
    
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            const auto& p = board[y][x];
            if (p.empty() || p.at("owner") != player) continue;

            string side = p.at("side");
            string orientation = "";
            if (side == "river" && p.count("orientation")) orientation = p.at("orientation");

            for (auto [dx,dy] : dirs) {
                int nx = x + dx;
                int ny = y + dy;
                if (!in_bounds(nx, ny, rows, cols)) continue;

                // same early block check as Python: block entering opponent score for the mover
                if (is_opponent_score_cell(nx, ny, player, rows, cols, score_cols)) continue;

                const auto& target = board[ny][nx];
                if (target.empty()) {
                    // plain move into empty square
                    moves.push_back(Move{{"move"}, {x,y}, {nx,ny}});
                } else {
                    string target_side = target.at("side");
                    string target_owner = target.at("owner");

                    if (target_side == "river") {
                        // follow river flow (note: python used player for flow here)
                        auto flow = get_river_flow_destinations(board, nx, ny, x, y, player, rows, cols, score_cols);
                        for (auto &f : flow) {
                            // python checks (again) not entering opponent score for final destination w.r.t mover
                            if (!is_opponent_score_cell(f.first, f.second, player, rows, cols, score_cols)) {
                                moves.push_back(Move{{"move"}, {x,y}, {f.first, f.second}});
                            }
                        }
                    } else {
                        int px = nx + dx;
                        int py = ny + dy;

                        // must be in bounds and the landing square empty
                        if (!in_bounds(px, py, rows, cols)) continue;
                        if (!board[py][px].empty()) continue;

                        // target_owner is the owner of the piece being pushed
                        // 1) don't push a piece that's currently sitting in *its own* scoring area
                        if (is_my_score_cell(nx, ny, target_owner, rows, cols, score_cols)) continue;

                        // 2) don't push a piece *into* that piece-owner's scoring area (don't help them)
                        if (is_my_score_cell(px, py, target_owner, rows, cols, score_cols)) continue;

                        // 3) additionally: don't push an opponent's piece into *your* scoring area
                        if (target_owner != player && is_my_score_cell(px, py, player, rows, cols, score_cols)) continue;

                        // passed all extra checks -> valid push (matches Python intent with added safety)
                        moves.push_back(Move{{"push"}, {x,y}, {nx,ny}, {px,py}});
                    }
                }
            } // dirs loop

            // Flip / rotate generation follows Python's behavior:
            if (side == "stone") {
                // Python temporarily treats the stone as a river with an orientation for move listing.
                // Add both orientation flips (do not mutate board here).
                moves.push_back(Move{{"flip"}, {x,y}, {}, {}, "horizontal"});
                moves.push_back(Move{{"flip"}, {x,y}, {}, {}, "vertical"});
            } else {
                // river piece: flip (to stone) and rotate (swap orientation)
                moves.push_back(Move{{"flip"}, {x,y}});
                // compute alternate orientation
                string new_ori = (orientation == "horizontal") ? "vertical" : "horizontal";
                moves.push_back(Move{{"rotate"}, {x,y}, {}, {}, new_ori});
                // Note: Python mutated p.orientation here; we do not mutate const board,
                // but we add the rotate move with the new orientation, matching the generated move list.
            }
        }
    }


    return moves;
}

// Main function
Board makemove(
    const Board& board,
    const Move& move,
    const string& player,
    int rows, int cols,
    const vector<int>& score_cols
) {
    auto new_board = board; // copy board

    if (move.action == "move") {
        int fx = move.from[0], fy = move.from[1];
        int tx = move.to[0], ty = move.to[1];
        
        if (!in_bounds(fx,fy,rows,cols) || !in_bounds(tx,ty,rows,cols)) return new_board;
        if (is_opponent_score_cell(tx,ty,player,rows,cols,score_cols)) return new_board;
        auto& piece = new_board[fy][fx];

        if (piece.empty() || piece.at("owner") != player) return new_board;
        
        if (new_board[ty][tx].empty()) {
            new_board[ty][tx] = piece;
            new_board[fy][fx].clear();
            return new_board;
        }

        int ptx = move.pushed_to[0], pty = move.pushed_to[1];
        int dx = tx - fx, dy = ty - fy;
        if (ptx != tx+dx || pty != ty+dy) return new_board;
        if (!in_bounds(ptx,pty,rows,cols)) return new_board;
        if (is_opponent_score_cell(ptx,pty,player,rows,cols,score_cols)) return new_board;
        if (!new_board[pty][ptx].empty()) return new_board;

        new_board[pty][ptx] = new_board[ty][tx];
        new_board[ty][tx] = piece;
        new_board[fy][fx].clear();
        return new_board;
    }

    else if (move.action == "push") {
        
        int fx = move.from[0], fy = move.from[1];
        int tx = move.to[0], ty = move.to[1];
        int px = move.pushed_to[0], py = move.pushed_to[1];

        if (!in_bounds(fx,fy,rows,cols) || !in_bounds(tx,ty,rows,cols) || !in_bounds(px,py,rows,cols))
            return new_board;

        auto& piece = new_board[fy][fx];
        if (piece.empty() || piece.at("owner") != player) return new_board;
        if (new_board[ty][tx].empty()) return new_board;
        if (!new_board[py][px].empty()) return new_board;
        if (piece.at("side")=="river" && new_board[ty][tx].at("side")=="river") return new_board;
        // cout << "inside push" << endl;
        
        if (new_board[ty][tx]["side"]=="stone" && new_board[fy][fx]["side"]=="stone"){
            new_board[py][px] = new_board[ty][tx];
            new_board[ty][tx] = piece;
            new_board[fy][fx].clear();
            return new_board;
        }

        // compute valid pushes
        auto valid_pairs = get_river_flow_destinations(new_board, fx, fy, fx, fy, player, rows, cols, score_cols);
        bool valid=false;
        for (auto& d : valid_pairs) {
            if (d.first==px && d.second==py) { valid=true; break; }
        }
        if (!valid) return new_board;

        new_board[py][px] = new_board[ty][tx];
        new_board[ty][tx] = piece;
        new_board[fy][fx].clear();
        
        if (new_board[ty][tx]["side"]=="river") {
            new_board[ty][tx]["side"]="stone";
            new_board[ty][tx].erase("orientation");
        }

        return new_board;
    }

    else if (move.action == "flip") {
        int fx = move.from[0], fy = move.from[1];
        if (!in_bounds(fx,fy,rows,cols)) return new_board;

        auto& piece = new_board[fy][fx];
        if (piece.empty() || piece.at("owner") != player) return new_board;

        if (piece.at("side")=="stone") {
            if (move.orientation != "horizontal" && move.orientation != "vertical") return new_board;
            piece["side"]="river";
            piece["orientation"]=move.orientation;

            auto flow = get_river_flow_destinations(new_board, fx, fy, fx, fy, player, rows, cols, score_cols);
            for (auto& d : flow) {
                if (is_opponent_score_cell(d.first,d.second,player,rows,cols,score_cols)) {
                    piece["side"]="stone";
                    piece.erase("orientation");
                    return new_board;
                }
            }
            piece["side"]="river";
            piece["orientation"]=move.orientation;
        } else {
            piece["side"]="stone";
            piece.erase("orientation");
        }
        return new_board;
    }

    else if (move.action == "rotate") {
        int fx = move.from[0], fy = move.from[1];
        if (!in_bounds(fx,fy,rows,cols)) return new_board;

        auto& piece = new_board[fy][fx];
        if (piece.empty() || piece.at("owner") != player) return new_board;
        if (piece.at("side")!="river") return new_board;

        piece["orientation"] = (piece["orientation"]=="vertical") ? "horizontal" : "vertical";

        auto flow = get_river_flow_destinations(new_board, fx, fy, fx, fy, player, rows, cols, score_cols);
        for (auto& d : flow) {
            if (is_opponent_score_cell(d.first,d.second,player,rows,cols,score_cols)) {
                piece["orientation"] = (piece["orientation"]=="vertical") ? "horizontal" : "vertical";
                return new_board;
            }
        }
        return new_board;
    }

    return new_board;
}


float EvalFunction() {
    // Use a random eval function for the sake of now, float: confirmed
    static random_device rd;
    static mt19937 gen(rd());
    uniform_real_distribution<float> dist(0.0, 100.0);

    return dist(gen);
}

int mhdistance(const map<string, string> &p, int x, int y, int row, const vector<int>& score_cols){
    // more close less score
    if (x>=score_cols[0] and x<=score_cols[score_cols.size()-1] and y==row){
        if (p.at("side")=="river"){
            return 1;
        }
        return 0;
    }
    // edit: put d1 in loop as we don't know how many score cols are there
    int d1 = 10000;
    for (auto col:score_cols){
        if (abs(x-col)<d1){
            d1 = abs(x-col);
        }
    }
        
    int d2=abs(y-row);
    int mh = d1+d2;
    return mh;
}

int mhdistance2(const map<string, string> &p, int x, int y, int row, const vector<int>& score_cols){
    // moving to a corner if there is high defense

    // if circle
    if (row==2){
        row=0;
    }
    else{
        row = 13; // stay in same row
    }
    int col = 0;
    
    int d1 = abs(x-col);
    int d2 = abs(y-row);
    int mh = d1+d2;
    return mh;
}

float EvalFunction2(const Board& board, string& rootplayer, const vector<int>& score_cols, float current_player_time) {
    // temporary evaluation function only using manhattan distance now
    float myscore = 0, oppscore = 0;
    int rows = board.cells.size();
    int cols = board.cells[0].size();
    int top_score_row = 2;
    int bottom_score_row = rows-3;
    int myscorerow,oppscorerow;
    if (rootplayer == "circle"){
        myscorerow = top_score_row;
        oppscorerow = bottom_score_row;
    }
    else{
        myscorerow = bottom_score_row;
        oppscorerow = top_score_row;
    }
    float scores[] = {2684354.56, 1342177.28, 671088.64, 335544.32, 167772.16, 83886.08, 41943.04, 20971.52, 10485.76, 5242.88, 2621.44, 1310.72, 655.36, 327.68, 163.84, 81.92, 40.96, 20.48, 10.24, 5.12, 2.56, 1.28, 0.64, 0.32, 0.16, 0.08, 0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.000525, 0.000425, 0.000325, 0.000225, 0.000125, 0.00011, 0.0001};
    // cout << "myplayer : " << rootplayer << endl;
    // cout << "myscorerow : " << myscorerow << endl;
    if (current_player_time<30 && current_player_time>=26 && get_score(board, rootplayer, rows, cols, score_cols)<2){
        // use mhdistance2
        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < cols; x++) {
                const auto& p = board[y][x];
                if (!p.empty()){
                    if (p.at("owner") == rootplayer){
                        
                        myscore+=scores[mhdistance2(p, x, y, myscorerow, score_cols)];
                        // cout << "my piece: " << p.at("side") << " at " << x << "," << y << " eval: " << mhdistance(p, x, y, myscorerow, score_cols) <<endl;
                    }
                    else{
                        oppscore+=scores[mhdistance2(p, x, y, oppscorerow, score_cols)];
                    }
                }
            }
        }
        return -myscore+oppscore;
    }

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            const auto& p = board[y][x];
            if (!p.empty()){
                if (p.at("owner") == rootplayer){
                    
                    myscore+=scores[mhdistance(p, x, y, myscorerow, score_cols)];
                    // cout << "my piece: " << p.at("side") << " at " << x << "," << y << " eval: " << mhdistance(p, x, y, myscorerow, score_cols) <<endl;
                }
                else{
                    oppscore+=scores[mhdistance(p, x, y, oppscorerow, score_cols)];
                }
            }
        }
    }
    return -myscore+oppscore;
}

float EvalFunction3(const Board& board, string& rootplayer, const vector<int>& score_cols) {
    // temporary evaluation function only using manhattan distance now
    float myscore = 0, oppscore = 0;
    int rows = board.cells.size();
    int cols = board.cells[0].size();
    int top_score_row = 2;
    int bottom_score_row = rows-3;
    int myscorerow,oppscorerow;
    if (rootplayer == "cicle"){
        myscorerow = top_score_row;
        oppscorerow = bottom_score_row;
    }
    else{
        myscorerow = bottom_score_row;
        oppscorerow = top_score_row;
    }
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            const auto& p = board[y][x];
            if (p.empty() || p.at("owner") == rootplayer){
                myscore+=mhdistance(p, x, y, myscorerow, score_cols);
            }
            else{
                oppscore+=mhdistance(p, x, y, oppscorerow, score_cols);
            }

        }
    }
    return myscore-oppscore;
}



struct StackFrame {
    Node* node;
    Board board;      // one board per level
    size_t childIdx;  // which child we’re exploring
};

Move minimax(Node* root, const Board& rootBoard, string& rootplayer,
             int rows, int cols, const vector<int>& score_cols, int maxDepth)
{
    // more close less score, hence opposite logic in minmax
    // cout << "inside minimax2" << endl;

    stack<StackFrame> stack;
    stack.push({root, rootBoard, 0});
    double val;
    root->score = (root->player != rootplayer ?
                   -numeric_limits<double>::infinity() :
                    numeric_limits<double>::infinity());

    Node *bestChild = nullptr, *parent_, *child, *node;

    while (!stack.empty()) {
        auto &frame = stack.top();
        node = frame.node;

        // Leaf or terminal node already handled
        if (node->children.empty()) {
            node->score = EvalFunction2(frame.board, rootplayer, score_cols,0);
            stack.pop();
            if (!stack.empty()) {
                parent_ = node->parent;
                if (parent_->player == rootplayer)
                    parent_->score = min(parent_->score, node->score);
                else
                    parent_->score = min(parent_->score, node->score);
            }
            continue;
        }

        // Optimization: If children are LEAVES (next depth == maxDepth)
        if (!node->children.empty() && node->children[0]->level == maxDepth) {
            double bestVal = (node->player != rootplayer ?
                                numeric_limits<double>::infinity() :
                               numeric_limits<double>::infinity());

            for (auto* child : node->children) {
                Board newBoard = makemove(frame.board, child->move,
                                          child->player, rows, cols, score_cols);
                val = EvalFunction2(newBoard, rootplayer, score_cols,0);
                // print_board(newBoard);  
                // cout << "eval function val for board is " << val << endl;

                child->score = val;
                if (node->player != rootplayer)
                    bestVal = min(bestVal, val);
                else
                    bestVal = min(bestVal, val);
            }

            node->score = bestVal;

            // Backpropagate
            stack.pop();
            if (!stack.empty()) {
                parent_ = node->parent;
                if (parent_->player != rootplayer)
                    parent_->score = min(parent_->score, node->score);
                else
                    parent_->score = min(parent_->score, node->score);
            }
            continue;
        }

        // Normal DFS expansion
        if (frame.childIdx < node->children.size()) {
            child = node->children[frame.childIdx++];
            Board newBoard = makemove(frame.board, child->move,
                                      child->player, rows, cols, score_cols);

            if (child->children.empty()) {
                // Skip pushing, will be handled above
            } else {
                child->score = (child->player != rootplayer ?
                                numeric_limits<double>::infinity() :
                                 numeric_limits<double>::infinity());
                stack.push({child, newBoard, 0});
            }
        } else {
            // All children done
            stack.pop();
            if (!stack.empty()) {
                parent_ = node->parent;
                if (parent_->player != rootplayer)
                    parent_->score = min(parent_->score, node->score);
                else
                    parent_->score = min(parent_->score, node->score);
            }
        }
    }

    // Pick best move
    float bestScore = numeric_limits<double>::infinity();
    for (auto child : root->children) {
        if (child->score < bestScore) {
            bestScore = child->score;
            bestChild = child;
        }
    }

    // cout << "minimax2 done with score: " << bestScore << endl;
    Move move = bestChild ? bestChild->move : Move();
    cout << "minimax player " << rootplayer <<" choosed move: " << move.action << " from " << move.from[0] << ", " << move.from[1] << " to " << move.to[0] << ", " << move.to[1] << " with score: " << bestScore << endl;
    return move;
}



Move minimax2(Node* root, const Board& rootBoard, string& rootplayer,
             int rows, int cols, const vector<int>& score_cols, int maxDepth, float current_player_time)
{
    // more close less score, hence opposite logic in minmax
    // cout << "inside minimax2" << endl;

    stack<StackFrame> stack;
    stack.push({root, rootBoard, 0});
    double val;
    root->score = (root->player != rootplayer ?
                   -numeric_limits<double>::infinity() :
                    numeric_limits<double>::infinity());

    Node *bestChild = nullptr, *parent_, *child, *node;

    while (!stack.empty()) {
        auto &frame = stack.top();
        node = frame.node;

        // Leaf or terminal node already handled
        if (node->children.empty()) {
            node->score = EvalFunction2(frame.board, rootplayer, score_cols, current_player_time);
            stack.pop();
            if (!stack.empty()) {
                parent_ = node->parent;
                if (parent_->player != rootplayer)
                    parent_->score = min(parent_->score, node->score);
                else
                    parent_->score = max(parent_->score, node->score);
            }
            continue;
        }

        // Optimization: If children are LEAVES (next depth == maxDepth)
        if (!node->children.empty() && node->children[0]->level == maxDepth) {
            double bestVal = (node->player != rootplayer ?
                                -numeric_limits<double>::infinity() :
                                numeric_limits<double>::infinity());

            for (auto* child : node->children) {
                Board newBoard = makemove(frame.board, child->move,
                                          child->player, rows, cols, score_cols);
                val = EvalFunction2(newBoard, rootplayer, score_cols, current_player_time);
                
                // print_board(newBoard);  
                // cout << "eval function val for player " << rootplayer << " is " << val << endl;

                child->score = val;
                if (node->player != rootplayer)
                    bestVal = max(bestVal, val);
                else
                    bestVal = min(bestVal, val);
            }

            node->score = bestVal;

            // Backpropagate
            stack.pop();
            if (!stack.empty()) {
                parent_ = node->parent;
                if (parent_->player != rootplayer)
                    parent_->score = max(parent_->score, node->score);
                else
                    parent_->score = min(parent_->score, node->score);
            }
            continue;
        }

        // Normal DFS expansion
        if (frame.childIdx < node->children.size()) {
            child = node->children[frame.childIdx++];
            Board newBoard = makemove(frame.board, child->move,
                                      child->player, rows, cols, score_cols);

            if (child->children.empty()) {
                // Skip pushing, will be handled above
            } else {
                child->score = (child->player != rootplayer ?
                                -numeric_limits<double>::infinity() :
                                 numeric_limits<double>::infinity());
                stack.push({child, newBoard, 0});
            }
        } else {
            // All children done
            stack.pop();
            if (!stack.empty()) {
                parent_ = node->parent;
                if (parent_->player != rootplayer)
                    parent_->score = max(parent_->score, node->score);
                else
                    parent_->score = min(parent_->score, node->score);
            }
        }
    }

    // Pick best move
    float bestScore = numeric_limits<double>::infinity();
    for (auto child : root->children) {
        if (child->score < bestScore) {
            bestScore = child->score;
            bestChild = child;
        }
    }

    // cout << "minimax2 done with score: " << bestScore << endl;
    Move move = bestChild ? bestChild->move : Move();
    // cout << "minimax player " << rootplayer <<" choosed move: " << move.action << " from " << move.from[0] << ", " << move.from[1] << " to " << move.to[0] << ", " << move.to[1] << " with score: " << bestScore << endl;
    return move;
}


Move IDS(Board& board, int rows, int cols, const vector<int>& score_cols, float current_player_time, float opponent_time, string player, int depth){
    /* Function that takes an input board and iteratively
    search through it's neigbhours upto some depth d
    */

    // cout << "score cols" << endl;

    Move move;

    // for(auto i=0;i<score_cols.size();i++){
    //     cout << score_cols[i] << endl;
    // }
    Node* root = new Node(0,player,0),current_node;
    // cout << " IDS of side " << player << endl;
    // q is current fringe, queue of leaf nodes.
    vector<Node*> q={root};
    // stack is current stack of boards, to use when we do traversal.
    Board new_board = board;
    vector<Board> stack = {board};
    vector<Move> moves;
    int qsize, leaves, c;
    float val;
    double rf;
    int top_score_row = 2;
    int bottom_score_row = rows-3;
    bool deff = false;
    int mydefenserow = player == "circle" ? bottom_score_row - 1 : top_score_row + 1;
    
    // cout << q.size() << " " << depth << endl;
    for(auto d=0;d<depth;d++){
        qsize = q.size();
        leaves = 0;
        // cout << "player " << player << " at level " << d+1 << endl;
        for(auto qi=0;qi<qsize;qi++){
            // temp code
            if (d>0){
                // creating a path from root to current node as we don't store states
                vector<Node*> path;
                Node* cur = q[qi];
                // cout << "working ..." << endl;
                while (cur->parent != nullptr) {
                    // cout << "inside parent search" << endl;
                    path.push_back(cur);
                    cur = cur->parent;
                }
                // cout << "working ..." << endl;
                reverse(path.begin(), path.end());
                new_board = board; // always start from the root board
                for (auto* node : path) {
                    new_board = makemove(new_board, node->move, node->player, rows, cols, score_cols);
                }
                

                // cout << "makemove about to do" << endl;
                // print_board(new_board);

                // cout << "makemovedone " << q[qi]->move.action << " at " <<  q[qi]->move.from[1] << "," << q[qi]->move.from[0] << " depth:" << d << endl;

            
                
                // cout << "makemove done" << endl;
            }


            // if (d>0){
            //     if (qi==0){
            //         stack.push_back(makemove(stack[d-1],q[qi]->move,q[qi]->player, row,col,score_cols));
            //         cout << "makemovedone1 " << q[qi]->move.action << " at " <<  q[qi]->move.from[1] << "," << q[qi]->move.from[0] << " depth:" << d << endl;
            //         print_board(stack[d]);                }
            //     else{
            //         stack[d] = makemove(stack[d-1],q[qi]->move,q[qi]->player, row,col,score_cols);
                    
            //         cout << "makemovedone2 " << q[qi]->move.action << " at " <<  q[qi]->move.from[1] << "," << q[qi]->move.from[0] << " depth:" << d << endl;
            //         print_board(stack[d]);
            //         // return move;
            //     }
            // }
            // moves = generate_all_moves(stack[d], row, col, score_cols, current_player_time, opponent_time, player);
            
            
            moves = generate_all_moves(new_board, rows, cols, score_cols, current_player_time, opponent_time, player);
            // print_board(new_board);
            // UndoInfo info = apply_move(board, q[qi]->move);
            // moves = generate_all_moves(board, row, col, score_cols, current_player_time, opponent_time, side);
            // cout << "local no of moves " << moves.size() << " at depth " << d << " for " << player << endl;

            // cout << "generatealldone" << endl;
            leaves+=moves.size();
            // this_thread::sleep_for(chrono::seconds(1));
            c=0;
            for(auto& move : moves){
                // checking defense is not moving
                for (int x = 0; x < int(cols/2); x++) {
                    if (move.from[1]==mydefenserow and move.from[0]==x+3){
                        deff = true;
                        break;
                    }
                }
                if (deff==true){
                    deff = false;
                    continue;
                }

                // if move from scoring area to outside don't consider it
                if (move.action == "move" || move.action == "push"){
                    int tx = move.to[0], ty = move.to[1], fx = move.from[0], fy = move.from[1];
                    
                    if (is_my_score_cell(fx,fy,player,rows,cols,score_cols) && !is_my_score_cell(tx,ty,player,rows,cols,score_cols)){
                        continue;
                    }
                    if (is_my_score_cell(fx,fy,player,rows,cols,score_cols) && is_my_score_cell(tx,ty,player,rows,cols,score_cols)){
                        // cout << "considering move/push within score cell" << endl;
                        if (randfgen()<0.9){
                            continue;
                        }
                    }

                    if (is_my_score_cell(tx,ty,player,rows,cols,score_cols) and !is_my_score_cell(fx,fy,player,rows,cols,score_cols) and move.action=="move"){
                        return move;
                    }
                    if (is_my_score_cell(tx,ty,player,rows,cols,score_cols) and move.action=="push"){
                        continue;
                    }
                }
                if (move.action == "flip"){
                    int fx = move.from[0], fy = move.from[1];
                    if (is_my_score_cell(fx,fy,player,rows,cols,score_cols) && new_board[fy][fx].at("side")=="river"){
                        return move;
                    }
                    else if (is_my_score_cell(fx,fy,player,rows,cols,score_cols) && new_board[fy][fx].at("side")=="stone"){
                        continue;
                    }
                }

                // if (move.action == "move" || move.action == "push"){
                //     cout << "1_considering " << move.action << " from " << move.from[0] << "," << move.from[1] << " to " << move.to[0] << "," << move.to[1] << endl;
                // }  
                // else{
                //     cout << "1_considering " << move.action << " at " << move.from[0] << "," << move.from[1] << endl;
                // } 

                // randomly ignore some moves to limit branching factor
                // rf = randfgen();
                // if (rf<0.4 and depth<3){
                //     continue;
                // }
                rf = randfgen();
                if (depth>=3){
                    // don't let push, only let river flow moves

                    if (move.action == "push" and rf<0.9){
                        continue;
                    }
                    if (move.action == "move" and rf<0.9){
                        int fx = move.from[0], fy = move.from[1], tx = move.to[0], ty = move.to[1];
                        if (abs(fx-tx)+abs(fy-ty)==1){
                            continue;
                        }
                    }
                    if (rf<0.6){
                        continue;
                    }
                }
                else{
                    // randomly ignore some moves to limit branching factor
                    rf = randfgen();
                    if (rf<0.1){
                        continue;
                    }
                }
                
                // add backward move or push with probability 0.5
                if (move.action == "move" || move.action == "push"){
                    // cout << "considering tactic 2" << endl;
                    rf = randfgen();
                    int fx = move.from[0], fy = move.from[1], tx = move.to[0], ty = move.to[1];
                    if (player=="circle" and fy<ty and ty>int(rows/2)){
                        // cout << "considering backward move for circle" << endl;
                        if (rf<0.8){
                            continue;
                        }
                    }
                    else if (player=="square" and fy>ty and ty<int(rows/2)){
                        // cout << "considering backward move for square" << endl;
                        if (rf<0.8){
                            continue;
                        }
                    }
                }

                Node* n = new Node(0,player,d+1,0.0,move);
                // if (move.action == "move" || move.action == "push"){
                //     cout << "2_considering " << move.action << " from " << move.from[0] << "," << move.from[1] << " to " << move.to[0] << "," << move.to[1] << endl;
                // }  
                // else{
                //     cout << "2_considering " << move.action << " at " << move.from[0] << "," << move.from[1] << endl;
                // } 
                // if (d+1==depth){
                    // if (c==0){
                    //     cout << player << ": ";
                    // }
                    // val = EvalFunction();
                    // n->score = val;
                    // cout << val << ",";
                // }
                n->parent = q[qi];
                q[qi]->children.push_back(n);
                q.push_back(n);
                c++;
                // if (c==5){
                //     c=0;
                //     break;
                // }
            }
            // undo_move(board, info);
            // cout << endl;
            
        }
        // cout << "ignored moves " << moves.size()-c << endl;
        // cout << "fringe size at depth " << d << " : " << q.size() << endl;
        for(auto qi=0;qi<qsize;qi++){
            q.erase(q.begin());
        }
        // cout << "total no of moves " << leaves << " at depth " << d << endl;
        if (randfgen()<0.5){
            player = (player == "square") ? "circle" : "square";  
        }
        // cout << "depth done ..." << d << endl;
    }
    // move = minimax(root, board, player, rows, cols, score_cols);
    move = minimax2(root, board, root->player, rows, cols, score_cols, depth, current_player_time);
    
    return move;
};



// ---- PyBind11 bindings ----
PYBIND11_MODULE(student_agent_module, m) {
    py::class_<Move>(m, "Move")
        .def_readonly("action", &Move::action)
        .def_readonly("from_pos", &Move::from)
        .def_readonly("to_pos", &Move::to)
        .def_readonly("pushed_to", &Move::pushed_to)
        .def_readonly("orientation", &Move::orientation);

    py::class_<StudentAgent>(m, "StudentAgent")
        .def(py::init<string>())
        .def("choose", &StudentAgent::choose);
}