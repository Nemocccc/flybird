#ifndef WALL_CONTROL_H
#define WALL_CONTROL_H

#include "windows_control.h"

class wall : public windows_control{
    protected:
        int wall_pos_x = 0;
        int wall_pos_y = 0;
        int gap = 9;
        int lower_boundry = 25;
        int upper_boundry = 1;

    public:
        wall(): wall(0, 0){};//默认构造函数
        wall(int x, int y);//构造函数a
        wall(const wall &w);//拷贝构造函数
        void wall_move();
        int wall_check();
        void wall_print();
        void boundry_print();
        int get_wall_x();
        int get_wall_y();
        int get_gap();
        int get_lower_boundry();
        int get_upper_boundry();
};

#endif