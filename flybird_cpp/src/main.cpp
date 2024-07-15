#include<iostream>
#include<windows.h>
#include "bird_control.h"
#include "windows_control.h"
#include "wall_control.h"

int main(){
    windows_control windows;
    windows.HideCursor();
    bird flybird;
    wall wall_obj[3] = {
        {40, 10},
        {60, 6},
        {80, 8}
    };

    while(!flybird.game_over(wall_obj[0].get_wall_x(), wall_obj[0].get_wall_y(), wall_obj[0].get_gap(), wall_obj[0].get_lower_boundry(), wall_obj[0].get_upper_boundry())){
        wall_obj[0].boundry_print();
        if(wall_obj[0].wall_check()){
            wall temp(wall_obj[0]);
            wall_obj[0] = wall_obj[1];
            wall_obj[1] = wall_obj[2];
            wall_obj[2] = temp;
        }
        wall_obj[0].wall_print();
        wall_obj[1].wall_print();
        wall_obj[2].wall_print();
        flybird.bird_show();

        Sleep(50);

        flybird.bird_move();
        for (int i = 0; i < 3; i++){
            wall_obj[i].wall_move();
        }
    }

    return 0;
}