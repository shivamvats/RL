#include "tic_tac_toe_env.h"
#include <iostream>

int main() {
    TicTacToeEnv env;
    env.printGame();
    env.makeMove(cross, 0);
    env.makeMove(cross, 3);
    env.makeMove(cross, 6);
    env.printGame();
    std::cout << "\n\n" << env.gameOver() << "\n";
    std::cout << env.isWinning(cross);
}
