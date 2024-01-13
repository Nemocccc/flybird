//gcc -version == 7.3.0

#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<conio.h>
#include<windows.h>

#define GAP 9
#define DIS 22

int upper_boundry;

typedef struct bird
{
    COORD pos;
    int score;
}BIRD;

void CheckWall(COORD wall[]); //显示柱子
void PrtBird(BIRD * bird); //显示小鸟
int CheckWin(COORD * wall, BIRD * bird);//检测小鸟是否碰墙或者超出上下边界。
void Begin(BIRD * bird);//显示上下边界和分数
BOOL SetConsoleColor(unsigned int wAttributes);//设置颜色
void Gotoxy(int x, int y);//定位光标
void HideCursor();//隐藏光标，减少闪屏

//思路：小鸟只走上下，柱子往左边走
int main()
{
    BIRD bird = {{22, 10}, 0};//小鸟的初始位置
    COORD wall[3] = {{40, 10}, {60, 6}, {80, 8}};//柱子的初始位置和高度
    int i;
    char ch;

    while (CheckWin(wall, &bird))//游戏循环
    {
        Begin(&bird);//刷新
        CheckWall(wall);//显示柱子
        PrtBird(&bird);//显示小鸟
        Sleep(200);

        if (kbhit())
        {
            ch = getch();
            if (ch == ' ')//用户输入空格
            {
                bird.pos.Y -= 1;//小鸟向上一格
            }
        }
        else
        {
            bird.pos.Y += 1;//小鸟向下一格
        }
        for(i = 0; i < 3; ++i)
        {
            wall[i].X--;//柱子向左一格
        }
    }

    return 0;
}

//函数功能：显示/刷新柱子
//参数：柱子位置的列表
void CheckWall(COORD wall[])
{
    int i;
    HideCursor();
    srand(time(NULL));//用时间作为随机数种子
    COORD temp = {wall[2].X + DIS, rand() % 13 + 5}//随机产生一个新的柱子

    if (wall[0].X < 10)//超出左边界
    {
        wall[0] = wall[1];//最左侧柱子消失，第二个柱子变成第一个
    }
}