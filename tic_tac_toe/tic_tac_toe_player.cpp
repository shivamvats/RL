#include "tic_tac_toe_player.h"
#include <algorithm>

TicTacToePlayer::TicTacToePlayer(Player sign) : sign_(sign) {}

std::vector<int> getEmptyLocations(std::vector<Player> grid,
                                   std::vector<int> &emptyLoc) {
    for (int i = 0; i < grid.size(); i++) {
        if (grid[i] == empty) emptyLoc.push_back(i);
    }
    return emptyLoc;
}

int TicTacToePlayerAI::getMove(TicTacToeEnv env) {
    std::vector<int> emptyLoc;
    getEmptyLocations(env.getGrid(), emptyLoc);

    return *std::min_element(emptyLoc.begin(), emptyLoc.end());
}
