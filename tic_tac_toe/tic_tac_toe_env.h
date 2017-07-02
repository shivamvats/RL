#ifndef TIC_TAC_TOE_ENV_H
#define TIC_TAC_TOE_ENV_H

#include <vector>

enum Player { nought = 0, cross, empty };

class TicTacToeEnv {
public:
    TicTacToeEnv();
    bool checkIfWinning(Player player);
    bool makeMove(Player player, int cell);
    bool isWinning(Player player);
    bool gameOver();
    Player winner();
    void printGame();

private:
    // 0, 1 are for players.
    // 2 is for empty cell.
    std::vector<Player> grid_;
};

#endif
