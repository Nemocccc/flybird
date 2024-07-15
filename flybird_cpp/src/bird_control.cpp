#include "bird_control.h"
#include <random>
#include <iostream>
#include <conio.h>

bird::bird(){//小鸟初始位置
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<int> dis(2, 20);

    x = 22;
    y = dis(gen);

    score = 0;
}

void bird::bird_show(){
    SetConsoleColor(0x0E);
    Gotoxy(x, y);
    std::cout << "O->" << std::endl;
    Gotoxy(0, 2);
    std::cout << "Score: " << score << std::endl;
}

void bird::bird_move(){
    if (kbhit()){
        char ch = getch();
        if (ch == ' ' || ch == 'w' || ch == 'W'){
            y -= 1;
        }
        else if (ch == 's' || ch == 'S'){
            y += 2;
        }
    }
    else {
        y += 1;
    }
}

int bird::game_over(int wall_x, int wall_y, int gap, int lower, int upper){
    if ((x >= wall_x && x <=wall_x + 5) && (y <= wall_y || y >= wall_y + gap)){
        return 1;
    }
    if (y <= upper || y >= lower){
        return 1;
    }

    score ++;
    return 0;
}