#include "tic_tac_toe_env.h"
#include <algorithm>
#include <iostream>

#define GRID_SIZE 9

TicTacToeEnv::TicTacToeEnv() : grid_(GRID_SIZE, empty) {}

bool TicTacToeEnv::makeMove(Player player, int cell) {
    if (grid_[cell] == empty) {
        grid_[cell] = player;
        return true;
    }
    return false;
}

bool TicTacToeEnv::isWinning(Player player) {
    // Horizontal cases
    for (int i = 0; i <= 6; i += 3) {
        if (grid_[i] == player && grid_[i + 1] == player &&
            grid_[i + 2] == player)
            return true;
    }

    // Vertical cases
    for (int i = 0; i <= 2; i++) {
        if (grid_[i] == player && grid_[i + 3] == player &&
            grid_[i + 6] == player)
            return true;
    }

    // Left-right diagonal
    if (grid_[0] == player && grid_[4] == player && grid_[8] == player)
        return true;
    // Right-left diagonal
    if (grid_[2] == player && grid_[4] == player && grid_[6] == player)
        return true;

    return false;
}

bool TicTacToeEnv::gameOver() {
    bool fullGrid = !std::any_of(grid_.begin(), grid_.end(),
                                 [&](Player p) { return p == empty; });
    if ((isWinning(Player::nought)) || isWinning(Player::cross) || fullGrid)
        return true;
    else
        return false;
}

Player TicTacToeEnv::winner() {
    if (isWinning(Player::nought))
        return nought;
    else if (isWinning(Player::cross))
        return Player::cross;
    else
        return Player::empty;
}

void TicTacToeEnv::printGame() {
    std::cout << grid_[0] << "|" << grid_[1] << "|" << grid_[2] << "\n";
    std::cout << "------\n";
    std::cout << grid_[3] << "|" << grid_[4] << "|" << grid_[5] << "\n";
    std::cout << "------\n";
    std::cout << grid_[6] << "|" << grid_[7] << "|" << grid_[8] << "\n";
}
