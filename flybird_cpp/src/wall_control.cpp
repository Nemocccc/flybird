#include "wall_control.h"
#include <iostream>
#include <time.h>
#include <stdlib.h>

wall::wall(int x, int y){
    wall_pos_x = x;
    wall_pos_y = y;
}

wall::wall(const wall &w){
    wall_pos_x = 80; //新横坐标为80
    wall_pos_y = w.wall_pos_y;
}

void wall::wall_move(){
    wall_pos_x --;
}

int wall::wall_check(){
    if (wall_pos_x <= 10){
        srand(time(NULL));
        wall_pos_x = 80;
        wall_pos_y = rand() % 13 + 5;
        return 1;
    }
    return 0;
}

void wall::boundry_print(){
    // 边界
    system("cls");
    Gotoxy(0, lower_boundry);
    std::cout << "============================================================================================================" << std::endl;
    Gotoxy(0, upper_boundry);
    std::cout << "============================================================================================================" << std::endl;
}

void wall::wall_print(){
    HideCursor();
    for (int i = 0; i < 3; i++){
        int j;
        SetConsoleColor(0x0C);
        for (j = 2; j < wall_pos_y; j++){
            Gotoxy(wall_pos_x + 1, j);
            std::cout << "#####" << std::endl;
        }
        Gotoxy(wall_pos_x, j);
        std::cout << " #####" << std::endl;
        // 下半部分柱子墙
        j += gap;
        Gotoxy(wall_pos_x, j);
        std::cout << " #####" << std::endl;
        j++;
        for (; j < 25; j++){
            Gotoxy(wall_pos_x + 1, j);
            std::cout << "#####" << std::endl;
        }
    }
}

int wall::get_wall_x(){
    return wall_pos_x;
}

int wall::get_wall_y(){
    return wall_pos_y;
}

int wall::get_gap(){
    return gap;
}

int wall::get_lower_boundry(){
    return lower_boundry;
}

int wall::get_upper_boundry(){
    return upper_boundry;
}