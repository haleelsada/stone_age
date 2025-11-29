#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <map>
#include <random>
#include <chrono>
#include <stack>

using namespace std;
#include <queue>
#include <set>
#include <algorithm>  // for find
#include <iostream>
namespace py = pybind11;


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
Move IDS(Board& board, int row, int col, const vector<int>& score_cols, float current_player_time, float opponent_time, string& side, int depth );
// ---- Declarations -----
vector<Move> generate_all_moves(const Board& board, int row, int col, const vector<int>& score_cols, float current_player_time, float opponent_time, const string& player);



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


// ---- Student Agent ----
class StudentAgent {
public:
    explicit StudentAgent(string side) : side(move(side)), gen(rd()) {}

    Move choose(const  vector<vector<map<string, string>>>& oldboard, int row, int col, const vector<int>& score_cols, float current_player_time, float opponent_time) {
        Board board = Board(oldboard);
        cout << "CPP Agent choosing move..." << endl;
        vector<Move> moves = generate_all_moves(board, row, col, score_cols, current_player_time, opponent_time, side);
        if (moves.empty()) {
            return {"move", {0,0}, {0,0}, {}, ""}; // fallback
        }
        Board board_copy = board;
        
        // calling IDS
        auto start = chrono::high_resolution_clock::now();
        // cout << "calling ids... " << endl;
        int depth = 1;
        // print_board(board);
        Move move = IDS(board_copy,row,col,score_cols,current_player_time,opponent_time,side,depth);
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        cout << "IDS took " << duration << " ms for depth: " << depth << endl;
        return move;
        // uniform_int_distribution<> dist(0, moves.size()-1);
        // return moves[dist(gen)];
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

vector<pair<int,int>> get_river_flow_destinations(
    const Board& board,
    int rx, int ry, int sx, int sy, const string& player,
    int rows, int cols, const vector<int>& score_cols,
    bool river_push=false
) {
    vector<pair<int,int>> destinations;
    set<pair<int,int>> visited;
    queue<pair<int,int>> q;
    q.push({rx, ry});

    while(!q.empty()) {
        auto [x, y] = q.front(); q.pop();
        if (visited.count({x,y}) || !in_bounds(x,y,rows,cols)) continue;
        visited.insert({x,y});

        auto cell = board[y][x];
        if (river_push && x==rx && y==ry) cell = board[sy][sx];

        if (cell.empty()) {
            if (!is_opponent_score_cell(x,y,player,rows,cols,score_cols)) {
                destinations.push_back({x,y});
            }
            continue;
        }

        if (cell.at("side") != "river") continue;

        vector<pair<int,int>> dirs;
        if (cell.at("orientation") == "horizontal") {
            dirs = {{1,0},{-1,0}};
        } else {
            dirs = {{0,1},{0,-1}};
        }

        for (auto [dx,dy] : dirs) {
            int nx = x+dx, ny = y+dy;
            while(in_bounds(nx,ny,rows,cols)) {
                if (is_opponent_score_cell(nx,ny,player,rows,cols,score_cols)) break;
                auto next_cell = board[ny][nx];
                if (nx==sx && ny==sy) { nx+=dx; ny+=dy; continue; }
                if (next_cell.empty()) {
                    destinations.push_back({nx,ny});
                    nx+=dx; ny+=dy; continue;
                }
                if (next_cell.at("side")=="river") { q.push({nx,ny}); break; }
                break;
            }
        }
    }

    set<pair<int,int>> seen;
    vector<pair<int,int>> out;
    for (auto& d : destinations) {
        if (!seen.count(d)) { seen.insert(d); out.push_back(d); }
    }
    return out;
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
    if (x>=score_cols[0] and x<=score_cols[score_cols.size()-1] and y==row){
        return -10;
    }
    
    int d1=abs(min({x-score_cols[0],x-score_cols[1],x-score_cols[2],x-score_cols[3]}));
    int d2=abs(y-row);

    return d1+d2;
}
float EvalFunction2(const Board& board, string& rootplayer, const vector<int>& score_cols) {
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
    return myscore;
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
    return myscore;
}



struct StackFrame {
    Node* node;
    Board board;      // one board per level
    size_t childIdx;  // which child we’re exploring
};

Move minimax(Node* root, const Board& rootBoard, string& rootplayer, int rows, int cols, const vector<int>& score_cols){
    // cout << "inside minimax" << endl;
    // using parent->player != rootplayer in logic cause at every node the player is the 
    // one who did last move, ie the level before it, but for minmax we are looking 
    // for opposite of this(imagine 2 level, at leaf player = me but in tree it will be opponent)
    // minimax2 showing better result
    stack<StackFrame> stack;

    stack.push({root, rootBoard, 0});
    root->score = (root->player != rootplayer ?
                   -numeric_limits<double>::infinity() :
                    numeric_limits<double>::infinity());

    Node *bestChild = nullptr, *parent_, *child, *node;
    
    while (!stack.empty()) {
        auto &frame = stack.top();
        node = frame.node;

        if (node->children.empty()) {
            // node->score = EvalFunction();
            stack.pop();

            if (!stack.empty()) {
                parent_ = node->parent;
                if (parent_->player != rootplayer){
                    parent_->score = max(parent_->score, node->score);
                    // cout << "max node choose at leaf:" << parent->score << endl;
                    // cout << "max node player:" << parent->player << ", root:" << rootplayer << endl;

                }
                else{
                    parent_->score = min(parent_->score, node->score);
                    // cout << "min node choose at leaf:" << parent->score << endl;
                }
            }
            continue;
        }

        // Still children left?
        if (frame.childIdx < node->children.size()) {
            child = node->children[frame.childIdx++];
            // Apply move starting from THIS node's board

            Board newBoard = makemove(frame.board, child->move, node->player, rows, cols, score_cols);

            if (child->children.empty()) {
                // Don't reset score if it's a leaf (already pre-evaluated)
                // Just push it to stack so it can backpropagate
            } else {
                // Internal node → initialize for minimax aggregation
                child->score = (child->player != rootplayer ?
                                -numeric_limits<double>::infinity() :
                                numeric_limits<double>::infinity());
            }
            
            // Push child level with its board
            stack.push({child, newBoard, 0});
        } else {
            // All children done, pop & backpropagate
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

    // Pick best child move
    float bestScore = -numeric_limits<double>::infinity();
    for (auto child : root->children) {
        if (child->score > bestScore) {
            bestScore = child->score;
            cout << child->score;
            bestChild = child;
        }
    }

    cout << " minimax over with score:" << bestScore  << endl;
    return bestChild ? bestChild->move : Move();
}



Move minimax2(Node* root, const Board& rootBoard, string& rootplayer,
             int rows, int cols, const vector<int>& score_cols, int maxDepth)
{
    cout << "inside minimax2" << endl;

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

        // Optimization: If children are LEAVES (next depth == maxDepth)
        if (!node->children.empty() && node->children[0]->level == maxDepth) {
            double bestVal = (node->player != rootplayer ?
                              -numeric_limits<double>::infinity() :
                               numeric_limits<double>::infinity());

            for (auto* child : node->children) {
                Board newBoard = makemove(frame.board, child->move,
                                          child->player, rows, cols, score_cols);
                val = EvalFunction2(newBoard, rootplayer, score_cols);
                // print_board(newBoard);  
                // cout << "eval function val for board is " << val << endl;

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
    float bestScore = -numeric_limits<double>::infinity();
    for (auto child : root->children) {
        if (child->score > bestScore) {
            bestScore = child->score;
            bestChild = child;
        }
    }

    // cout << "minimax2 done with score: " << bestScore << endl;
    return bestChild ? bestChild->move : Move();
}


Move IDS(Board& board, int rows, int cols, const vector<int>& score_cols, float current_player_time, float opponent_time, string& player, int depth){
    /* Function that takes an input board and iteratively
    search through it's neigbhours upto some depth d
    */

    // cout << "score cols" << endl;

    Move move;

    // for(auto i=0;i<score_cols.size();i++){
    //     cout << score_cols[i] << endl;
    // }
    Node* root = new Node(0,player,0),current_node;
    // cout << "this is side " << player << endl;
    // q is current fringe, queue of leaf nodes.
    vector<Node*> q={root};
    // stack is current stack of boards, to use when we do traversal.
    Board new_board = board;
    vector<Board> stack = {board};
    vector<Move> moves;
    int qsize, leaves, c;
    float val;
    // cout << q.size() << " " << depth << endl;
    for(auto d=0;d<depth;d++){
        qsize = q.size();
        leaves = 0;
        cout << "player; " << player << "at level " << d+1 << endl;
        for(auto qi=0;qi<qsize;qi++){
            // temp code
            if (d>0){
                
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

            // UndoInfo info = apply_move(board, q[qi]->move);
            // moves = generate_all_moves(board, row, col, score_cols, current_player_time, opponent_time, side);
            // cout << "local no of moves " << moves.size() << " at depth " << d << " for " << player << endl;

            // cout << "generatealldone" << endl;
            leaves+=moves.size();
            // this_thread::sleep_for(chrono::seconds(1));
            c=0;
            for(auto& move : moves){
                Node* n = new Node(0,player,d+1,0.0,move);

                if (d+1==depth){
                    // if (c==0){
                    //     cout << player << ": ";
                    // }
                    // val = EvalFunction();
                    // n->score = val;
                    // cout << val << ",";
                }
                n->parent = q[qi];
                q[qi]->children.push_back(n);
                q.push_back(n);
                // c++;
                // if (c==5){
                //     c=0;
                //     break;
                // }
            }
            // undo_move(board, info);
            // cout << endl;
            
        }
        // cout << "fringe size at depth " << d << " : " << qsize << endl;
        for(auto qi=0;qi<qsize;qi++){
            q.erase(q.begin());
        }
        // cout << "total no of moves " << leaves << " at depth " << d << endl;
        player = (player == "square") ? "circle" : "square";
        // cout << "depth done ..." << d << endl;
    }
    // move = minimax(root, board, player, rows, cols, score_cols);
    move = minimax2(root, board, player, rows, cols, score_cols, depth);
    
    return move;
};

vector<Move> generate_all_moves(
    const Board& board,
    int rows, int cols,
    const vector<int>& score_cols,
    float current_player_time,
    float opponent_time,
    const string& player
) {
    vector<Move> moves;


    vector<pair<int,int>> dirs = {{1,0},{-1,0},{0,1},{0,-1}};
    // cout << "generating moves for "<< player << endl;

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
                if (!in_bounds(nx,ny,rows,cols)) continue;
                if (is_opponent_score_cell(nx,ny,player,rows,cols,score_cols)) continue;

                const auto& target = board[ny][nx];
                if (target.empty() && !is_opponent_score_cell(nx,ny,player, rows, cols, score_cols)) {
                    // cout << "move from " << x << "," << y << " to " << nx << "," << ny << endl;
                    moves.push_back(Move{{"move"}, {x,y}, {nx,ny}});
                } else {
                    string target_side = target.at("side");
                    string target_owner = target.at("owner");

                    if (target_side == "river") {
                        auto flow = get_river_flow_destinations(board, nx, ny, x, y, player, rows, cols, score_cols);
                        for (auto [fx,fy] : flow) {
                            if (!is_opponent_score_cell(fx,fy,player, rows, cols, score_cols)){
                                // cout << "river move from " << x << "," << y << " to " << nx << "," << ny << endl;
                                moves.push_back(Move{{"move"}, {x,y}, {fx,fy}});
                            }
                        }
                    } else {
                        int px = nx + dx;
                        int py = ny + dy;
                        if (in_bounds(px,py,rows,cols) && board[py][px].empty() &&
                            !is_opponent_score_cell(px,py,target_owner,rows,cols,score_cols) &&
                            !is_opponent_score_cell(nx,ny,target_owner,rows,cols,score_cols)) {
                            moves.push_back(Move{{"push"}, {x,y}, {nx,ny}, {px,py}});
                        }
                    }
                }
            }

            // Flip and Rotate moves
            if (side == "stone") {
                for (auto ori : {"horizontal", "vertical"}) {
                    moves.push_back(Move{{"flip"}, {x,y}, {}, {}, ori});
                }
            } else {
                moves.push_back(Move{{"flip"}, {x,y}});
                string new_ori = (orientation=="horizontal") ? "vertical" : "horizontal";
                moves.push_back(Move{{"rotate"}, {x,y}, {}, {}, new_ori});
            }
        }
    }

    return moves;
}

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