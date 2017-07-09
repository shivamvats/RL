#include "tic_tac_toe_env.h"

class TicTacToePlayer {
public:
    TicTacToePlayer();
    TicTacToePlayer(Player sign);
    Player getSign() { return sign_; }
    virtual int getMove(TicTacToeEnv env) = 0;

private:
    Player sign_;
};

// Always plays the smallest empty location as its next move.
class TicTacToePlayerAI : public TicTacToePlayer {
public:
    TicTacToePlayerAI(Player sign) : TicTacToePlayer(sign) {}
    int getMove(TicTacToeEnv env);
};

// class TicTacToePlayerTest : TicTacToePlayer {
//    TicTacToePlayerTest();
//};
