#ifndef BIRD_CONTROL_H
#define BIRD_CONTROL_H

#include "wall_control.h"

class bird : public wall{
    public:
        bird();
        void bird_show();
        void bird_move();
        int game_over(int wall_x, int wall_y, int gap, int lower, int upper);

    protected:
        int x = 0;
        int y = 0;
        int score = 0;
};

#endif