#include "windows_control.h"
#include <conio.h>
#include<windows.h>

void windows_control::HideCursor(){
    HANDLE handle = GetStdHandle (STD_OUTPUT_HANDLE);
    CONSOLE_CURSOR_INFO CursorInfo ;
    GetConsoleCursorInfo (handle, &CursorInfo);//获取控制台光标信息
    CursorInfo .bVisible = 0 ;//隐藏控制台光标
    SetConsoleCursorInfo (handle, &CursorInfo);//设置控制台光标状态
}

int windows_control::SetConsoleColor(unsigned int wAttributes){
    HANDLE houtput = GetStdHandle(STD_OUTPUT_HANDLE);
    if (houtput == INVALID_HANDLE_VALUE){
        return -1;
    }
    return SetConsoleTextAttribute(houtput, wAttributes);
}

void windows_control::Gotoxy(int x, int y){
    COORD pos = {x, y};
    HANDLE hOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleCursorPosition(hOutput, pos);
}