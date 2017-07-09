#include <iostream>

#include "tic_tac_toe_env.h"
#include "tic_tac_toe_player.h"

int main() {
    TicTacToeEnv env;
    env.printGame();

    TicTacToePlayerAI ai(cross);
    TicTacToePlayerAI human(nought);

    int i = 0;
    int j = 0;
    while (!env.gameOver()) {
        if (i % 2 == 0) {
            j = env.makeMove(cross, ai.getMove(env));
            // std::cout << j << "\n";
        } else {
            j = env.makeMove(nought, human.getMove(env));
            // std::cout << j << "\n";
        }
        env.printGame();
        i++;
    }
    // env.makeMove(cross, 3);
    // env.makeMove(cross, 6);
    env.printGame();
}
